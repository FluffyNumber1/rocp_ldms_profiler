#include "config.hpp"
#include "device.hpp"
#include "pc.hpp"
#include "publisher.hpp"
#include "trace.hpp"
#include "diagnostics.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/callback_tracing.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <unistd.h>

namespace rocm_ldms {
namespace {

std::string hostname() {
  std::array<char, 256> buffer{};
  if (::gethostname(buffer.data(), buffer.size() - 1) == 0)
    return buffer.data();
  return "unknown";
}

} // namespace

/* Coordinates the independently selectable PC, device, and trace services. */
class RocprofilerTool {
public:
  bool initialize() {
    DiagnosticTimer timer{"initialize_ns"};
    const char *path = std::getenv("ROCM_LDMS_CONFIG");
    if (!path || !*path) {
      std::cerr << "rocprof_ldms: ROCM_LDMS_CONFIG is not set\n";
      return false;
    }

    try {
      config_ = load_config(path);
    } catch (const std::exception &error) {
      std::cerr << "rocprof_ldms: " << error.what() << '\n';
      return false;
    }

    if (!config_.pc.enabled && !config_.device.enabled &&
        !config_.trace.enabled && !config_.tool.control_only) {
      std::cerr << "rocprof_ldms: enable at least one collection mode\n";
      return false;
    }

    producer_ = hostname();
    const char *flux_job = std::getenv("FLUX_JOB_ID");
    job_id_ = flux_job && *flux_job ? flux_job : "none";
    pid_ = static_cast<uint64_t>(::getpid());

    if (!publisher_.initialize(config_.ldms, config_.tool.output))
      return false;
    if (config_.pc.enabled && !pc_.configure(config_.pc))
      return false;
    if (config_.device.enabled && !device_.configure(config_.device))
      return false;
    if (config_.trace.enabled && !trace_.configure(config_.trace))
      return false;
    if (config_.device.enabled) {
      // Device counting needs HSA's profiling queues. During tool_init the
      // runtime is not ready yet; configure now, start on its notification.
      auto status = rocprofiler_create_context(&runtime_context_);
      if (status == ROCPROFILER_STATUS_SUCCESS) {
        const rocprofiler_tracing_operation_t operation =
            ROCPROFILER_RUNTIME_INITIALIZATION_HSA;
        status = rocprofiler_configure_callback_tracing_service(
            runtime_context_, ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION,
            &operation, 1, runtime_initialized, this);
      }
      if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "rocprof_ldms: runtime callback configuration failed with "
                  << "ROCProfiler status " << status << '\n';
        return false;
      }
    }
    return true;
  }

  bool start() {
    DiagnosticTimer timer{"start_ns"};
    // Start tracing first so it is active before any other enabled service.
    if (config_.trace.enabled && !trace_.start())
      return false;
    if (config_.pc.enabled && !pc_.start())
      return false;
    timer_stopping_ = false;
    timer_thread_ = std::thread{&RocprofilerTool::timer_loop, this};
    if (config_.device.enabled) {
      const auto status = rocprofiler_start_context(runtime_context_);
      if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "rocprof_ldms: runtime callback start failed with "
                  << "ROCProfiler status " << status << '\n';
        return false;
      }
      runtime_started_ = true;
      std::clog << "rocprof_ldms: device collection armed; waiting for HSA initialization\n";
    } else {
      std::clog << "rocprof_ldms: collection started\n";
    }
    return true;
  }

  void shutdown() {
    DiagnosticTimer timer{"shutdown_ns"};
    {
      // Serialize against a deferred device start before tearing down buffers.
      std::lock_guard<std::mutex> lock{runtime_mutex_};
      shutting_down_ = true;
    }
    if (runtime_started_) {
      // Finalization may have already unregistered the callback context.
      // As with the PC/trace services, only stop a context that still exists
      // and is active. Preserve diagnostics for unexpected failures.
      int active = 0;
      auto status = rocprofiler_context_is_active(runtime_context_, &active);
      if (status == ROCPROFILER_STATUS_SUCCESS && active)
        status = rocprofiler_stop_context(runtime_context_);
      if (status != ROCPROFILER_STATUS_SUCCESS &&
          status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
        std::cerr << "rocprof_ldms: runtime callback stop failed with "
                  << "ROCProfiler status " << status << '\n';
      runtime_started_ = false;
    }
    stop_timer();

    if (config_.device.enabled)
      device_.shutdown();
    if (config_.pc.enabled)
      pc_.shutdown();
    if (config_.trace.enabled)
      trace_.shutdown();

    publish_windows(true);
  }

private:
  static void runtime_initialized(rocprofiler_callback_tracing_record_t record,
                                  rocprofiler_user_data_t *, void *data) {
    if (!data ||
        record.kind != ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION ||
        record.operation != ROCPROFILER_RUNTIME_INITIALIZATION_HSA)
      return;
    auto *self = static_cast<RocprofilerTool *>(data);
    std::lock_guard<std::mutex> lock{self->runtime_mutex_};
    if (self->shutting_down_ || self->device_start_attempted_)
      return;
    self->device_start_attempted_ = true;
    if (self->device_.start())
      std::clog << "rocprof_ldms: collection started\n";
    else
      std::cerr << "rocprof_ldms: device collection failed after HSA initialization\n";
  }

  bool publish_windows(bool final_window = false) {
    DiagnosticTimer timer{final_window ? "final_window_ns" : "periodic_window_ns"};
    bool success = true;
    const size_t limit = config_.tool.max_records_per_message;

    if (config_.pc.enabled &&
        !pc_.publish_window(publisher_, kPcMessageTag, producer_, job_id_, pid_,
                            limit)) {
      std::cerr << "rocprof_ldms: failed to publish "
                << (final_window ? "final " : "") << "PC window\n";
      success = false;
    }
    if (config_.device.enabled &&
        !device_.publish_window(publisher_, kDeviceMessageTag, producer_,
                                job_id_, pid_, limit)) {
      std::cerr << "rocprof_ldms: failed to publish "
                << (final_window ? "final " : "") << "device window\n";
      success = false;
    }
    if (config_.trace.enabled &&
        !trace_.publish_window(publisher_, kTraceMessageTag, producer_, job_id_,
                               pid_, limit)) {
      std::cerr << "rocprof_ldms: failed to publish "
                << (final_window ? "final " : "") << "trace window\n";
      success = false;
    }
    return success;
  }

  void timer_loop() {
    const auto interval = std::chrono::milliseconds{config_.tool.window_ms};
    auto next = std::chrono::steady_clock::now() + interval;
    std::unique_lock<std::mutex> lock{timer_mutex_};

    while (!timer_stopping_) {
      if (timer_cv_.wait_until(lock, next, [this] { return timer_stopping_; }))
        break;

      lock.unlock();
      publish_windows();
      lock.lock();

      next += interval;
      const auto now = std::chrono::steady_clock::now();
      if (next <= now)
        next = now + interval;
    }
  }

  void stop_timer() {
    {
      std::lock_guard<std::mutex> lock{timer_mutex_};
      timer_stopping_ = true;
    }
    timer_cv_.notify_all();
    if (timer_thread_.joinable())
      timer_thread_.join();
  }

  ToolConfig config_{};
  Publisher publisher_{};
  PcSampler pc_{};
  DeviceSampler device_{};
  TraceSampler trace_{};
  rocprofiler_context_id_t runtime_context_{};
  std::mutex runtime_mutex_{};
  bool runtime_started_ = false;
  bool device_start_attempted_ = false;
  bool shutting_down_ = false;

  std::string producer_{};
  std::string job_id_{};
  uint64_t pid_ = 0;

  std::thread timer_thread_{};
  std::mutex timer_mutex_{};
  std::condition_variable timer_cv_{};
  bool timer_stopping_ = false;
};

std::unique_ptr<RocprofilerTool> g_tool;

int tool_init(rocprofiler_client_finalize_t, void *) {
  g_tool = std::make_unique<RocprofilerTool>();
  if (!g_tool->initialize() || !g_tool->start()) {
    g_tool->shutdown();
    g_tool.reset();
    Diagnostics::get().save();
    return -1;
  }
  return 0;
}

void tool_fini(void *) {
  if (g_tool) {
    g_tool->shutdown();
    g_tool.reset();
    Diagnostics::get().save();
  }
}

} // namespace rocm_ldms

extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *client_id) {
  client_id->name = "ROCm_LDMS_Tool";
  static rocprofiler_tool_configure_result_t configuration = {
      sizeof(rocprofiler_tool_configure_result_t), rocm_ldms::tool_init,
      rocm_ldms::tool_fini, nullptr};
  return &configuration;
}
