#pragma once

#include "config.hpp"
#include "publisher.hpp"
#include "window.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocm_ldms {

/* One long-format row for one device-counter instance. */
struct DeviceRecord {
  uint64_t agent_id = 0;
  uint64_t sample_timestamp = 0;
  uint64_t counter_instance_id = 0;
  uint64_t counter_id = 0;
  uint64_t dispatch_id = 0;
  std::string counter_name{};
  uint64_t value_valid = 1;
  double value = 0.0;
};

class DeviceSampler {
public:
  bool configure(const ToolConfig::Device &config);
  bool start();
  void shutdown();

  bool publish_window(Publisher &publisher, const std::string &tag,
                      const std::string &producer, const std::string &job_id,
                      uint64_t pid, size_t max_records);

private:
  struct AgentState {
    DeviceSampler *owner = nullptr;
    rocprofiler_agent_id_t id{};
    std::string name{};
    rocprofiler_context_id_t context{};
    rocprofiler_buffer_id_t buffer{};
    rocprofiler_callback_thread_t callback_thread{};
    rocprofiler_counter_config_id_t profile{};
    std::unordered_map<uint64_t, std::string> counter_names{};
    bool context_created = false;
    bool buffer_created = false;
    bool profile_created = false;
    bool running = false;
    size_t ordinal = 0;
    uint32_t node_id = 0;
  };

  static rocprofiler_status_t
  collect_gpu_agents(rocprofiler_agent_version_t version, const void **agents,
                     size_t count, void *user_data);

  static rocprofiler_status_t
  collect_supported_counters(rocprofiler_agent_id_t agent_id,
                             rocprofiler_counter_id_t *counters, size_t count,
                             void *user_data);

  static void set_profile(rocprofiler_context_id_t context_id,
                          rocprofiler_agent_id_t agent_id,
                          rocprofiler_device_counting_agent_cb_t set_config,
                          void *user_data);

  static void buffer_callback(rocprofiler_context_id_t context_id,
                              rocprofiler_buffer_id_t buffer_id,
                              rocprofiler_record_header_t **headers,
                              size_t count, void *user_data,
                              uint64_t drop_count);

  bool configure_agent(AgentState &agent);
  void sampling_loop();
  void stop_sampling_thread();

  ToolConfig::Device config_{};
  std::vector<std::unique_ptr<AgentState>> agents_{};
  Window<DeviceRecord> window_{};

  std::thread sampling_thread_{};
  std::mutex sampling_mutex_{};
  std::condition_variable sampling_cv_{};
  bool sampling_stopping_ = false;
};

} // namespace rocm_ldms
