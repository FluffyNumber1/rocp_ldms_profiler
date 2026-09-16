#pragma once
#include <vector>
#include <string>
#include <cstddef>


namespace rocm_ldms{
    struct ToolConfig{
        struct Tool{
            bool control_only = false;
            std::string output = "ldms"; // ldms, serialize, discard (device only)
            size_t window_ms = 1000;
            // 256 keeps every LDMS 4.5 message below its 1 MiB limit.
            size_t max_records_per_message = 256;
        } tool;
         struct Ldms
        {
            std::string host = "localhost";
            std::string port = "411";
            std::string xprt = "sock";
            std::string auth = "none";
        } ldms;

        struct Pc
        {
            bool enabled = false;
            size_t interval_cycles = 1u << 20;
        } pc;

        struct Device
        {
            std::string agents = "all"; // GPU ordinals in SDK enumeration, not HIP remapped indices
            bool flush_on_window = false;
            bool enabled = false;
            size_t interval_ms = 50;
            std::vector<std::string> counters{};
        } device;

        struct Trace
        {
            bool enabled = false;
            bool hip_runtime = true;
            bool kernel_dispatch = true;
            bool memory_copy = true;
            bool memory_allocation = true;
            bool scratch_memory = true;
            bool runtime_initialization = true;
            bool correlation_retirement = true;
        } trace;
    };
    ToolConfig load_config(const std::string& path);
} //namespace rocm_ldms
