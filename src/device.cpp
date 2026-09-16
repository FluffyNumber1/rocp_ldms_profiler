#include "device.hpp"
#include "diagnostics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace rocm_ldms {
namespace {

constexpr size_t kBufferSize = 1024 * 1024;
constexpr size_t kWatermark = kBufferSize / 2;

bool check_status(rocprofiler_status_t status, const char *operation) {
  if (status == ROCPROFILER_STATUS_SUCCESS)
    return true;

  std::cerr << "rocprof_ldms: " << operation
            << " failed with ROCProfiler status " << status;
  if (status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)
    std::cerr << " (ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)";
  std::cerr << '\n';
  return false;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::string json_escape(const std::string &value) {
  std::string output;
  output.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
    case '"':
      output += "\\\"";
      break;
    case '\\':
      output += "\\\\";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      output += ch;
      break;
    }
  }
  return output;
}

void write_record(std::ostringstream &out, const DeviceRecord &record) {
  // U64 values travel as strings because OVIS JSON integers are signed.
  out << '{' << "\"agent_id\":\"" << record.agent_id << "\","
      << "\"sample_timestamp\":\"" << record.sample_timestamp << "\","
      << "\"counter_instance_id\":\"" << record.counter_instance_id << "\","
      << "\"counter_id\":\"" << record.counter_id << "\","
      << "\"dispatch_id\":\"" << record.dispatch_id << "\","
      << "\"counter_name\":\"" << json_escape(record.counter_name) << "\","
      << "\"value_valid\":\"" << record.value_valid << "\","
      << "\"value\":" << std::setprecision(17) << record.value << '}';
}

} // namespace

rocprofiler_status_t
DeviceSampler::collect_gpu_agents(rocprofiler_agent_version_t version,
                                  const void **agents, size_t count,
                                  void *user_data) {
  if (version != ROCPROFILER_AGENT_INFO_VERSION_0 || !user_data)
    return ROCPROFILER_STATUS_ERROR;

  auto *self = static_cast<DeviceSampler *>(user_data);
  auto *agent_list = reinterpret_cast<const rocprofiler_agent_t **>(agents);

  for (size_t i = 0; i < count; ++i) {
    if (agent_list[i]->type != ROCPROFILER_AGENT_TYPE_GPU)
      continue;

    auto state = std::make_unique<AgentState>();
    state->owner = self;
    state->id = agent_list[i]->id;
    state->ordinal = self->agents_.size();
    state->node_id = agent_list[i]->node_id;
    state->name = agent_list[i]->name ? agent_list[i]->name : "unknown";
    self->agents_.emplace_back(std::move(state));
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
DeviceSampler::collect_supported_counters(rocprofiler_agent_id_t,
                                          rocprofiler_counter_id_t *counters,
                                          size_t count, void *user_data) {
  if (!user_data)
    return ROCPROFILER_STATUS_ERROR;
  auto *output =
      static_cast<std::vector<rocprofiler_counter_id_t> *>(user_data);
  output->insert(output->end(), counters, counters + count);
  return ROCPROFILER_STATUS_SUCCESS;
}

void DeviceSampler::set_profile(
    rocprofiler_context_id_t context_id, rocprofiler_agent_id_t agent_id,
    rocprofiler_device_counting_agent_cb_t set_config, void *user_data) {
  auto *agent = static_cast<AgentState *>(user_data);
  if (!agent || agent->id.handle != agent_id.handle)
    return;

  const auto status = set_config(context_id, agent->profile);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::cerr << "rocprof_ldms: setting device profile for GPU '" << agent->name
              << "' failed with ROCProfiler status " << status << '\n';
  }
}

bool DeviceSampler::configure_agent(AgentState &agent) {
  std::vector<rocprofiler_counter_id_t> supported;
  if (!check_status(rocprofiler_iterate_agent_supported_counters(
                        agent.id, collect_supported_counters, &supported),
                    "rocprofiler_iterate_agent_supported_counters (device)"))
    return false;

  std::unordered_set<std::string> requested(config_.counters.begin(),
                                            config_.counters.end());
  std::vector<rocprofiler_counter_id_t> selected;

  for (const auto counter : supported) {
    rocprofiler_counter_info_v0_t info{};
    if (rocprofiler_query_counter_info(counter,
                                       ROCPROFILER_COUNTER_INFO_VERSION_0,
                                       &info) != ROCPROFILER_STATUS_SUCCESS ||
        !info.name)
      continue;

    if (requested.erase(info.name) != 0) {
      selected.push_back(counter);
      agent.counter_names.emplace(counter.handle, info.name);
      // Optional characterization artifact, separate from the LDMS wire schema.
      if (const char *path = std::getenv("ROCM_LDMS_COUNTER_METADATA")) {
        std::ofstream metadata(path, std::ios::app);
        metadata << "{\"agent_id\":\"" << agent.id.handle << "\",\"sdk_ordinal\":"
                 << agent.ordinal << ",\"node_id\":" << agent.node_id
                 << ",\"counter_id\":\"" << counter.handle
                 << "\",\"name\":\"" << json_escape(info.name)
                 << "\",\"description\":\"" << json_escape(info.description ? info.description : "")
                 << "\",\"block\":\"" << json_escape(info.block ? info.block : "")
                 << "\",\"expression\":\"" << json_escape(info.expression ? info.expression : "")
                 << "\",\"is_derived\":" << (info.is_derived ? "true" : "false");
        rocprofiler_counter_info_v1_t dimensions{};
        dimensions.size = sizeof(dimensions);
        const auto dimension_status = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_1, &dimensions);
        metadata << ",\"dimensions_available\":"
                 << (dimension_status == ROCPROFILER_STATUS_SUCCESS ? "true" : "false")
                 << ",\"instances\":[";
        if (dimension_status == ROCPROFILER_STATUS_SUCCESS) {
          for (uint64_t i = 0; i < dimensions.dimensions_instances_count; ++i) {
            const auto *instance = dimensions.dimensions_instances[i];
            if (i) metadata << ',';
            metadata << "{\"id\":\"" << instance->instance_id << "\",\"dimensions\":[";
            for (uint64_t j = 0; j < instance->dimensions_count; ++j) {
              const auto *dim = instance->dimensions[j];
              if (j) metadata << ',';
              metadata << "{\"name\":\"" << json_escape(dim->dimension_name ? dim->dimension_name : "")
                       << "\",\"index\":" << dim->index << '}';
            }
            metadata << "]}";
          }
        }
        metadata << "]}\n";
        if (!metadata) std::cerr << "rocprof_ldms: counter metadata write failed\n";
      }
    }
  }

  if (!requested.empty()) {
    std::cerr << "rocprof_ldms: GPU '" << agent.name
              << "' does not support requested device counter(s): ";
    bool first = true;
    for (const auto &name : requested) {
      std::cerr << (first ? "" : ", ") << name;
      first = false;
    }
    std::cerr << '\n';
    return false;
  }

  if (!check_status(rocprofiler_create_context(&agent.context),
                    "rocprofiler_create_context (device)"))
    return false;
  agent.context_created = true;

  if (!check_status(
          rocprofiler_create_buffer(agent.context, kBufferSize, kWatermark,
                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                    buffer_callback, &agent, &agent.buffer),
          "rocprofiler_create_buffer (device)"))
    return false;
  agent.buffer_created = true;

  if (!check_status(rocprofiler_create_callback_thread(&agent.callback_thread),
                    "rocprofiler_create_callback_thread (device)") ||
      !check_status(rocprofiler_assign_callback_thread(agent.buffer,
                                                       agent.callback_thread),
                    "rocprofiler_assign_callback_thread (device)"))
    return false;

  if (!check_status(rocprofiler_create_counter_config(agent.id, selected.data(),
                                                      selected.size(),
                                                      &agent.profile),
                    "rocprofiler_create_counter_config (device)"))
    return false;
  agent.profile_created = true;

  if (!check_status(
          rocprofiler_configure_device_counting_service(
              agent.context, agent.buffer, agent.id, set_profile, &agent),
          "rocprofiler_configure_device_counting_service"))
    return false;

  int valid = 0;
  return check_status(rocprofiler_context_is_valid(agent.context, &valid),
                      "rocprofiler_context_is_valid (device)") &&
         valid != 0;
}

bool DeviceSampler::configure(const ToolConfig::Device &config) {
  config_ = config;
  window_.reset();
  agents_.clear();

  if (!check_status(rocprofiler_query_available_agents(
                        ROCPROFILER_AGENT_INFO_VERSION_0, collect_gpu_agents,
                        sizeof(rocprofiler_agent_t), this),
                    "rocprofiler_query_available_agents (device)"))
    return false;

  if (agents_.empty()) {
    std::cerr << "rocprof_ldms: no GPU agents found for device counters\n";
    return false;
  }

  if (config_.agents != "all") {
    std::unordered_set<size_t> selected;
    std::istringstream input{config_.agents};
    std::string item;
    try {
      while (std::getline(input, item, ',')) {
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos)
          throw std::invalid_argument{"ordinal"};
        const auto ordinal = std::stoull(item);
        if (ordinal >= agents_.size() || !selected.insert(ordinal).second)
          throw std::invalid_argument{"ordinal"};
      }
      if (selected.empty()) throw std::invalid_argument{"empty"};
    } catch (const std::exception &) {
      std::cerr << "rocprof_ldms: invalid device.agents selection: " << config_.agents << '\n';
      return false;
    }
    agents_.erase(std::remove_if(agents_.begin(), agents_.end(),
        [&](const auto &agent) { return !selected.count(agent->ordinal); }), agents_.end());
  }

  for (auto &agent : agents_) {
    if (!configure_agent(*agent))
      return false;
  }
  return true;
}

bool DeviceSampler::start() {
  DiagnosticTimer timer{"device_start_ns"};
  for (auto &agent : agents_) {
    if (!check_status(rocprofiler_start_context(agent->context),
                      "rocprofiler_start_context (device)")) {
      shutdown();
      return false;
    }
    agent->running = true;
    std::clog << "rocprof_ldms: device counters enabled on GPU '" << agent->name
              << "' every " << config_.interval_ms << " ms; SDK ordinal=" << agent->ordinal
              << " agent_id=" << agent->id.handle << '\n';
  }

  sampling_stopping_ = false;
  sampling_thread_ = std::thread{&DeviceSampler::sampling_loop, this};
  return true;
}

void DeviceSampler::sampling_loop() {
  const auto interval = std::chrono::milliseconds{config_.interval_ms};
  auto next = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock{sampling_mutex_};

  while (!sampling_stopping_) {
    lock.unlock();
    for (auto &agent : agents_) {
      if (!agent->running)
        continue;
      rocprofiler_user_data_t user_data{};
      user_data.value = now_ns();
      DiagnosticTimer timer{"sample_ns"};
      const auto status = rocprofiler_sample_device_counting_service(
          agent->context, user_data, ROCPROFILER_COUNTER_FLAG_NONE,
          nullptr, nullptr);
      // SDK finalization can begin before tool_fini joins this thread.
      // Once finalized, no subsequent agent reads can succeed. End the loop
      // without treating that terminal lifecycle state as a sampling failure.
      if (status == ROCPROFILER_STATUS_ERROR_FINALIZED) {
        std::clog << "rocprof_ldms: device sampling stopped: SDK finalized\n";
        return;
      }
      check_status(status, "rocprofiler_sample_device_counting_service");
      Diagnostics::get().add(status == ROCPROFILER_STATUS_SUCCESS ? "sample_success" : "sample_errors", 1);
    }
    lock.lock();

    next += interval;
    if (next <= std::chrono::steady_clock::now())
      next = std::chrono::steady_clock::now() + interval;
    sampling_cv_.wait_until(lock, next, [this] { return sampling_stopping_; });
  }
}

void DeviceSampler::stop_sampling_thread() {
  {
    std::lock_guard<std::mutex> lock{sampling_mutex_};
    sampling_stopping_ = true;
  }
  sampling_cv_.notify_all();
  if (sampling_thread_.joinable())
    sampling_thread_.join();
}

void DeviceSampler::shutdown() {
  DiagnosticTimer timer{"device_shutdown_ns"};
  stop_sampling_thread();

  for (auto &agent : agents_) {
    if (agent->running) {
      int active = 0;
      const auto query = rocprofiler_context_is_active(agent->context, &active);
      if (query == ROCPROFILER_STATUS_SUCCESS && active)
        check_status(rocprofiler_stop_context(agent->context),
                     "rocprofiler_stop_context (device)");
      agent->running = false;
    }
    if (agent->buffer_created) {
      check_status(rocprofiler_flush_buffer(agent->buffer),
                   "rocprofiler_flush_buffer (device)");
      check_status(rocprofiler_destroy_buffer(agent->buffer),
                   "rocprofiler_destroy_buffer (device)");
      agent->buffer_created = false;
    }
    if (agent->profile_created) {
      check_status(rocprofiler_destroy_counter_config(agent->profile),
                   "rocprofiler_destroy_counter_config (device)");
      agent->profile_created = false;
    }
  }
}

void DeviceSampler::buffer_callback(rocprofiler_context_id_t,
                                    rocprofiler_buffer_id_t,
                                    rocprofiler_record_header_t **headers,
                                    size_t count, void *user_data,
                                    uint64_t drop_count) {
  auto *agent = static_cast<AgentState *>(user_data);
  if (!agent || !agent->owner)
    return;
  DiagnosticTimer timer{"device_callback_ns"};
  Diagnostics::get().add("buffer_drops", drop_count);
  size_t accepted_records = 0;

  if (drop_count)
    std::cerr << "rocprof_ldms: ROCProfiler device buffer dropped "
              << drop_count << " records\n";

  for (size_t i = 0; i < count; ++i) {
    const auto *header = headers[i];
    if (!header ||
        header->hash != rocprofiler_record_header_compute_hash(header->category,
                                                               header->kind) ||
        header->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS ||
        header->kind != ROCPROFILER_COUNTER_RECORD_VALUE)
      continue;

    const auto *source =
        static_cast<const rocprofiler_counter_record_t *>(header->payload);
    if (!source)
      continue;

    rocprofiler_counter_id_t counter_id{};
    if (rocprofiler_query_record_counter_id(source->id, &counter_id) !=
        ROCPROFILER_STATUS_SUCCESS)
      continue;

    DeviceRecord record{};
    record.agent_id = source->agent_id.handle;
    record.sample_timestamp = source->user_data.value;
    record.counter_instance_id = source->id;
    record.counter_id = counter_id.handle;
    record.dispatch_id = source->dispatch_id;
    record.value_valid = std::isfinite(source->counter_value) ? 1 : 0;
    record.value = record.value_valid ? source->counter_value : 0.0;

    const auto name = agent->counter_names.find(counter_id.handle);
    record.counter_name =
        name == agent->counter_names.end() ? "unknown" : name->second;
    agent->owner->window_.add(std::move(record));
    ++accepted_records;
  }
  Diagnostics::get().add("callback_records", accepted_records);
}

bool DeviceSampler::publish_window(Publisher &publisher, const std::string &tag,
                                   const std::string &producer,
                                   const std::string &job_id, uint64_t pid,
                                   size_t max_records) {
  // The SDK buffer and our publication window are separate queues. Test this
  // explicitly: the historical behavior only flushes at watermark/shutdown.
  if (config_.flush_on_window) {
    DiagnosticTimer timer{"flush_ns"};
    for (auto &agent : agents_) {
      if (agent->buffer_created) {
        const auto status = rocprofiler_flush_buffer(agent->buffer);
        if (status == ROCPROFILER_STATUS_ERROR_FINALIZED) break;
        check_status(status, "rocprofiler_flush_buffer (periodic device)");
      }
    }
  }
  auto snapshot = window_.rotate();
  if (snapshot.records.empty())
    return true;
  if (max_records == 0)
    return false;

  const size_t chunk_count =
      (snapshot.records.size() + max_records - 1) / max_records;
  Diagnostics::get().add("window_records", snapshot.records.size());
  Diagnostics::get().window(snapshot.sequence, snapshot.records.size(), chunk_count);
  if (publisher.discard()) return true;
  DiagnosticTimer timer{"device_encode_publish_ns"};
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t begin = chunk * max_records;
    const size_t end = std::min(begin + max_records, snapshot.records.size());
    std::ostringstream out;
    out << '{' << "\"schema\":\"rocm_device_raw_window\","
        << "\"producer\":\"" << json_escape(producer) << "\","
        << "\"job_id\":\"" << json_escape(job_id) << "\","
        << "\"pid\":" << pid << ','
        << "\"window_start_ns\":" << snapshot.start_ns << ','
        << "\"window_end_ns\":" << snapshot.end_ns << ','
        << "\"window_sequence\":" << snapshot.sequence << ','
        << "\"chunk_index\":" << chunk << ','
        << "\"chunk_count\":" << chunk_count << ','
        << "\"records_max_len\":" << max_records << ',' << "\"records\":[";
    for (size_t i = begin; i < end; ++i) {
      if (i != begin)
        out << ',';
      write_record(out, snapshot.records[i]);
    }
    out << "]}";
    if (!publisher.publish(tag, out.str()))
      return false;
  }
  return true;
}

} // namespace rocm_ldms
