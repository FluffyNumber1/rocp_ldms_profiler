#pragma once

#include "config.hpp"
#include "publisher.hpp"
#include "window.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace rocm_ldms {

/*
 * One normalized tracing row. valid_fields identifies which optional columns
 * are meaningful for a particular ROCProfiler buffer-tracing kind.
 */
struct TraceRecord {
  uint64_t valid_fields = 0;
  uint64_t kind = 0;
  uint64_t operation = 0;
  std::string kind_name{};
  std::string operation_name{};

  uint64_t start_timestamp = 0;
  uint64_t end_timestamp = 0;
  uint64_t correlation_internal = 0;
  uint64_t correlation_external = 0;
  uint64_t thread_id = 0;

  uint64_t agent_id = 0;
  uint64_t queue_id = 0;
  uint64_t kernel_id = 0;
  uint64_t dispatch_id = 0;
  uint64_t src_agent_id = 0;
  uint64_t dst_agent_id = 0;

  uint64_t bytes = 0;
  uint64_t address = 0;
  uint64_t src_address = 0;
  uint64_t dst_address = 0;
  uint64_t flags = 0;
  uint64_t version = 0;
  uint64_t instance = 0;

  uint64_t private_segment_size = 0;
  uint64_t group_segment_size = 0;
  uint64_t workgroup_x = 0;
  uint64_t workgroup_y = 0;
  uint64_t workgroup_z = 0;
  uint64_t grid_x = 0;
  uint64_t grid_y = 0;
  uint64_t grid_z = 0;
};

enum TraceValidField : uint64_t {
  TRACE_HAS_TIMESTAMPS = 1ull << 0,
  TRACE_HAS_CORRELATION = 1ull << 1,
  TRACE_HAS_THREAD = 1ull << 2,
  TRACE_HAS_AGENT = 1ull << 3,
  TRACE_HAS_QUEUE = 1ull << 4,
  TRACE_HAS_KERNEL = 1ull << 5,
  TRACE_HAS_DISPATCH = 1ull << 6,
  TRACE_HAS_SRC_AGENT = 1ull << 7,
  TRACE_HAS_DST_AGENT = 1ull << 8,
  TRACE_HAS_BYTES = 1ull << 9,
  TRACE_HAS_ADDRESS = 1ull << 10,
  TRACE_HAS_SRC_ADDRESS = 1ull << 11,
  TRACE_HAS_DST_ADDRESS = 1ull << 12,
  TRACE_HAS_FLAGS = 1ull << 13,
  TRACE_HAS_VERSION = 1ull << 14,
  TRACE_HAS_INSTANCE = 1ull << 15,
  TRACE_HAS_DISPATCH_GEOMETRY = 1ull << 16,
};

class TraceSampler {
public:
  bool configure(const ToolConfig::Trace &config);
  bool start();
  void shutdown();

  bool publish_window(Publisher &publisher, const std::string &tag,
                      const std::string &producer, const std::string &job_id,
                      uint64_t pid, size_t max_records);

private:
  static void buffer_callback(rocprofiler_context_id_t context_id,
                              rocprofiler_buffer_id_t buffer_id,
                              rocprofiler_record_header_t **headers,
                              size_t count, void *user_data,
                              uint64_t drop_count);

  static std::string kind_name(rocprofiler_buffer_tracing_kind_t kind);
  static std::string operation_name(rocprofiler_buffer_tracing_kind_t kind,
                                    rocprofiler_tracing_operation_t operation);
  static TraceRecord common_record(rocprofiler_buffer_tracing_kind_t kind,
                                   rocprofiler_tracing_operation_t operation);

  bool configure_kind(rocprofiler_buffer_tracing_kind_t kind);
  void handle_header(const rocprofiler_record_header_t &header);

  rocprofiler_context_id_t context_{};
  rocprofiler_buffer_id_t buffer_{};
  rocprofiler_callback_thread_t callback_thread_{};
  Window<TraceRecord> window_{};
  bool context_created_ = false;
  bool buffer_created_ = false;
  bool running_ = false;
};

} // namespace rocm_ldms
