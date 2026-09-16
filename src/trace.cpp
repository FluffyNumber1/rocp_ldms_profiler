#include "trace.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace rocm_ldms {
namespace {

constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr size_t kWatermark = kBufferSize / 2;

bool check_status(rocprofiler_status_t status, const char *operation) {
  if (status == ROCPROFILER_STATUS_SUCCESS)
    return true;
  std::cerr << "rocprof_ldms: " << operation
            << " failed with ROCProfiler status " << status << '\n';
  return false;
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

void write_u64(std::ostringstream &out, const char *name, uint64_t value) {
  out << '"' << name << "\":\"" << value << '\"';
}

void write_record(std::ostringstream &out, const TraceRecord &record) {
  out << '{';
  write_u64(out, "valid_fields", record.valid_fields);
  out << ',';
  write_u64(out, "kind", record.kind);
  out << ',';
  write_u64(out, "operation", record.operation);
  out << ',';
  out << "\"kind_name\":\"" << json_escape(record.kind_name) << "\",";
  out << "\"operation_name\":\"" << json_escape(record.operation_name) << "\",";
  write_u64(out, "start_timestamp", record.start_timestamp);
  out << ',';
  write_u64(out, "end_timestamp", record.end_timestamp);
  out << ',';
  write_u64(out, "correlation_internal", record.correlation_internal);
  out << ',';
  write_u64(out, "correlation_external", record.correlation_external);
  out << ',';
  write_u64(out, "thread_id", record.thread_id);
  out << ',';
  write_u64(out, "agent_id", record.agent_id);
  out << ',';
  write_u64(out, "queue_id", record.queue_id);
  out << ',';
  write_u64(out, "kernel_id", record.kernel_id);
  out << ',';
  write_u64(out, "dispatch_id", record.dispatch_id);
  out << ',';
  write_u64(out, "src_agent_id", record.src_agent_id);
  out << ',';
  write_u64(out, "dst_agent_id", record.dst_agent_id);
  out << ',';
  write_u64(out, "bytes", record.bytes);
  out << ',';
  write_u64(out, "address", record.address);
  out << ',';
  write_u64(out, "src_address", record.src_address);
  out << ',';
  write_u64(out, "dst_address", record.dst_address);
  out << ',';
  write_u64(out, "flags", record.flags);
  out << ',';
  write_u64(out, "version", record.version);
  out << ',';
  write_u64(out, "instance", record.instance);
  out << ',';
  write_u64(out, "private_segment_size", record.private_segment_size);
  out << ',';
  write_u64(out, "group_segment_size", record.group_segment_size);
  out << ',';
  write_u64(out, "workgroup_x", record.workgroup_x);
  out << ',';
  write_u64(out, "workgroup_y", record.workgroup_y);
  out << ',';
  write_u64(out, "workgroup_z", record.workgroup_z);
  out << ',';
  write_u64(out, "grid_x", record.grid_x);
  out << ',';
  write_u64(out, "grid_y", record.grid_y);
  out << ',';
  write_u64(out, "grid_z", record.grid_z);
  out << '}';
}

template <typename Correlation>
void set_correlation(TraceRecord &output, const Correlation &correlation) {
  output.correlation_internal = correlation.internal;
  output.correlation_external = correlation.external.value;
  output.valid_fields |= TRACE_HAS_CORRELATION;
}

} // namespace

std::string TraceSampler::kind_name(rocprofiler_buffer_tracing_kind_t kind) {
  const char *name = nullptr;
  if (rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr) ==
          ROCPROFILER_STATUS_SUCCESS &&
      name)
    return name;
  return "unknown";
}

std::string
TraceSampler::operation_name(rocprofiler_buffer_tracing_kind_t kind,
                             rocprofiler_tracing_operation_t operation) {
  const char *name = nullptr;
  if (rocprofiler_query_buffer_tracing_kind_operation_name(
          kind, operation, &name, nullptr) == ROCPROFILER_STATUS_SUCCESS &&
      name)
    return name;
  return "unknown";
}

TraceRecord
TraceSampler::common_record(rocprofiler_buffer_tracing_kind_t kind,
                            rocprofiler_tracing_operation_t operation) {
  TraceRecord output{};
  output.kind = static_cast<uint64_t>(kind);
  output.operation = static_cast<uint64_t>(operation);
  output.kind_name = kind_name(kind);
  output.operation_name = operation_name(kind, operation);
  return output;
}

bool TraceSampler::configure_kind(rocprofiler_buffer_tracing_kind_t kind) {
  return check_status(rocprofiler_configure_buffer_tracing_service(
                          context_, kind, nullptr, 0, buffer_),
                      "rocprofiler_configure_buffer_tracing_service");
}

bool TraceSampler::configure(const ToolConfig::Trace &config) {
  window_.reset();
  if (!check_status(rocprofiler_create_context(&context_),
                    "rocprofiler_create_context (trace)"))
    return false;
  context_created_ = true;

  if (!check_status(
          rocprofiler_create_buffer(context_, kBufferSize, kWatermark,
                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                    buffer_callback, this, &buffer_),
          "rocprofiler_create_buffer (trace)"))
    return false;
  buffer_created_ = true;

  std::vector<rocprofiler_buffer_tracing_kind_t> kinds;
  if (config.hip_runtime)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API);
  if (config.kernel_dispatch)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);
  if (config.memory_copy)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY);
  if (config.memory_allocation)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION);
  if (config.scratch_memory)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY);
  if (config.runtime_initialization)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION);
  if (config.correlation_retirement)
    kinds.push_back(ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT);

  if (kinds.empty()) {
    std::cerr
        << "rocprof_ldms: trace is enabled but no trace kinds are enabled\n";
    return false;
  }

  for (const auto kind : kinds) {
    if (!configure_kind(kind))
      return false;
  }

  if (!check_status(rocprofiler_create_callback_thread(&callback_thread_),
                    "rocprofiler_create_callback_thread (trace)") ||
      !check_status(
          rocprofiler_assign_callback_thread(buffer_, callback_thread_),
          "rocprofiler_assign_callback_thread (trace)"))
    return false;

  int valid = 0;
  return check_status(rocprofiler_context_is_valid(context_, &valid),
                      "rocprofiler_context_is_valid (trace)") &&
         valid != 0;
}

bool TraceSampler::start() {
  if (!context_created_ || running_)
    return context_created_;
  if (!check_status(rocprofiler_start_context(context_),
                    "rocprofiler_start_context (trace)"))
    return false;
  running_ = true;
  std::clog << "rocprof_ldms: buffered tracing enabled\n";
  return true;
}

void TraceSampler::shutdown() {
  if (running_) {
    int active = 0;
    const auto query = rocprofiler_context_is_active(context_, &active);
    if (query == ROCPROFILER_STATUS_SUCCESS && active)
      check_status(rocprofiler_stop_context(context_),
                   "rocprofiler_stop_context (trace)");
    else if (query != ROCPROFILER_STATUS_SUCCESS &&
             query != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
      check_status(query, "rocprofiler_context_is_active (trace)");
    running_ = false;
  }

  if (buffer_created_) {
    check_status(rocprofiler_flush_buffer(buffer_),
                 "rocprofiler_flush_buffer (trace)");
    check_status(rocprofiler_destroy_buffer(buffer_),
                 "rocprofiler_destroy_buffer (trace)");
    buffer_created_ = false;
  }
}

void TraceSampler::handle_header(const rocprofiler_record_header_t &header) {
  const auto kind = static_cast<rocprofiler_buffer_tracing_kind_t>(header.kind);
  TraceRecord output{};

  switch (kind) {
  case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API: {
    const auto *record =
        static_cast<const rocprofiler_buffer_tracing_hip_api_record_t *>(
            header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->start_timestamp;
    output.end_timestamp = record->end_timestamp;
    output.thread_id = record->thread_id;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: {
    const auto *record = static_cast<
        const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
        header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->start_timestamp;
    output.end_timestamp = record->end_timestamp;
    output.thread_id = record->thread_id;
    output.agent_id = record->dispatch_info.agent_id.handle;
    output.queue_id = record->dispatch_info.queue_id.handle;
    output.kernel_id = record->dispatch_info.kernel_id;
    output.dispatch_id = record->dispatch_info.dispatch_id;
    output.private_segment_size = record->dispatch_info.private_segment_size;
    output.group_segment_size = record->dispatch_info.group_segment_size;
    output.workgroup_x = record->dispatch_info.workgroup_size.x;
    output.workgroup_y = record->dispatch_info.workgroup_size.y;
    output.workgroup_z = record->dispatch_info.workgroup_size.z;
    output.grid_x = record->dispatch_info.grid_size.x;
    output.grid_y = record->dispatch_info.grid_size.y;
    output.grid_z = record->dispatch_info.grid_size.z;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD |
                           TRACE_HAS_AGENT | TRACE_HAS_QUEUE |
                           TRACE_HAS_KERNEL | TRACE_HAS_DISPATCH |
                           TRACE_HAS_DISPATCH_GEOMETRY;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: {
    const auto *record =
        static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t *>(
            header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->start_timestamp;
    output.end_timestamp = record->end_timestamp;
    output.thread_id = record->thread_id;
    output.src_agent_id = record->src_agent_id.handle;
    output.dst_agent_id = record->dst_agent_id.handle;
    output.bytes = record->bytes;
    output.src_address = record->src_address.value;
    output.dst_address = record->dst_address.value;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD |
                           TRACE_HAS_SRC_AGENT | TRACE_HAS_DST_AGENT |
                           TRACE_HAS_BYTES | TRACE_HAS_SRC_ADDRESS |
                           TRACE_HAS_DST_ADDRESS;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION: {
    const auto *record = static_cast<
        const rocprofiler_buffer_tracing_memory_allocation_record_t *>(
        header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->start_timestamp;
    output.end_timestamp = record->end_timestamp;
    output.thread_id = record->thread_id;
    output.agent_id = record->agent_id.handle;
    output.address = record->address.value;
    output.bytes = record->allocation_size;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD |
                           TRACE_HAS_AGENT | TRACE_HAS_ADDRESS |
                           TRACE_HAS_BYTES;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY: {
    const auto *record =
        static_cast<const rocprofiler_buffer_tracing_scratch_memory_record_t *>(
            header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->start_timestamp;
    output.end_timestamp = record->end_timestamp;
    output.thread_id = record->thread_id;
    output.agent_id = record->agent_id.handle;
    output.queue_id = record->queue_id.handle;
    output.flags = record->flags;
    output.bytes = record->allocation_size;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD |
                           TRACE_HAS_AGENT | TRACE_HAS_QUEUE | TRACE_HAS_FLAGS |
                           TRACE_HAS_BYTES;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION: {
    const auto *record = static_cast<
        const rocprofiler_buffer_tracing_runtime_initialization_record_t *>(
        header.payload);
    if (!record)
      return;
    output = common_record(kind, record->operation);
    output.start_timestamp = record->timestamp;
    output.end_timestamp = record->timestamp;
    output.thread_id = record->thread_id;
    output.version = record->version;
    output.instance = record->instance;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_THREAD |
                           TRACE_HAS_VERSION | TRACE_HAS_INSTANCE;
    set_correlation(output, record->correlation_id);
    break;
  }
  case ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT: {
    const auto *record = static_cast<
        const rocprofiler_buffer_tracing_correlation_id_retirement_record_t *>(
        header.payload);
    if (!record)
      return;
    output = common_record(kind, 0);
    output.start_timestamp = record->timestamp;
    output.end_timestamp = record->timestamp;
    output.correlation_internal = record->internal_correlation_id;
    output.valid_fields |= TRACE_HAS_TIMESTAMPS | TRACE_HAS_CORRELATION;
    break;
  }
  default:
    return;
  }

  window_.add(std::move(output));
}

void TraceSampler::buffer_callback(rocprofiler_context_id_t,
                                   rocprofiler_buffer_id_t,
                                   rocprofiler_record_header_t **headers,
                                   size_t count, void *user_data,
                                   uint64_t drop_count) {
  auto *self = static_cast<TraceSampler *>(user_data);
  if (!self)
    return;
  if (drop_count)
    std::cerr << "rocprof_ldms: ROCProfiler trace buffer dropped " << drop_count
              << " records\n";

  for (size_t i = 0; i < count; ++i) {
    const auto *header = headers[i];
    if (!header ||
        header->hash != rocprofiler_record_header_compute_hash(header->category,
                                                               header->kind) ||
        header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING)
      continue;
    self->handle_header(*header);
  }
}

bool TraceSampler::publish_window(Publisher &publisher, const std::string &tag,
                                  const std::string &producer,
                                  const std::string &job_id, uint64_t pid,
                                  size_t max_records) {
  auto snapshot = window_.rotate();
  if (snapshot.records.empty())
    return true;
  if (max_records == 0)
    return false;

  const size_t chunk_count =
      (snapshot.records.size() + max_records - 1) / max_records;
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t begin = chunk * max_records;
    const size_t end = std::min(begin + max_records, snapshot.records.size());
    std::ostringstream out;
    out << '{' << "\"schema\":\"rocm_trace_raw_window\","
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
