#pragma once

#include "config.hpp"

extern "C"{
    #include <ldms/ldms.h>
}

#include <string>

/*
 * Shared LDMS Message Service publisher.
 *
 * Owns one LDMS transport connection. Collection services provide a
 * message tag and JSON payload; Publisher sends the message to LDMS.
 */


namespace rocm_ldms{
    inline constexpr const char* kPcMessageTag = "rocm_pc_raw";
    inline constexpr const char* kDeviceMessageTag = "rocm_device_raw";
    inline constexpr const char* kTraceMessageTag = "rocm_trace_raw";
   class Publisher
    {
    public:
        ~Publisher();

        bool initialize(const ToolConfig::Ldms& config, const std::string& output = "ldms");
        bool discard() const { return output_ == "discard"; }

        bool publish(const std::string& tag, const std::string& json);

    private:
        bool connect();
        void close();

        ldms_t transport_ = nullptr;

        std::string xprt_{};
        std::string host_{};
        std::string port_{};
        std::string auth_{};
        std::string output_ = "ldms";
        const uint32_t kMessagePermissions = 0444; //Means the owner, group, and others can all read the published messgae. Permission bits to decide policy.
    };
}//namespace rocm_ldms
