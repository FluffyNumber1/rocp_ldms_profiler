#include "publisher.hpp"
#include "diagnostics.hpp"

#include <iostream>

namespace rocm_ldms{
    bool Publisher::initialize(const ToolConfig::Ldms& config, const std::string& output){
        output_ = output;
        xprt_ = config.xprt;
        host_ = config.host;
        port_ = config.port;
        auth_ = config.auth;
        return output_ != "ldms" || connect();
    }
    void Publisher::close(){
        if(transport_ ==nullptr){
            return;
        }
        ldms_xprt_close(transport_);
        ldms_xprt_put(transport_, "rocprof_ldms_tool");
        transport_ = nullptr;
    }
    bool Publisher::publish(const std::string& tag, const std::string& json)
    {
        DiagnosticTimer timer{"publish_ns"};
        Diagnostics::get().add("serialized_bytes", json.size() + 1);
        if (output_ != "ldms") return true;
        // Reconnect only if a previous failure cleared the transport.
        if(transport_ == nullptr && !connect())
            return false;

        const int rc = ldms_msg_publish(
            transport_,
            tag.c_str(),
            LDMS_MSG_JSON,
            nullptr, // Use process UID/GID.
            kMessagePermissions,
            json.c_str(),
            json.size() + 1);

        if(rc != 0)
        {
            std::cerr
                << "Failed to publish ROCProfiler message to tag: "
                << tag
                << std::endl;

            // Consider the transport bad after a failed publish.
            // The next publish() call will reconnect.
            close();
            return false;
        }
        return true;
    }

    bool Publisher::connect(){
        DiagnosticTimer timer{"connect_ns"};
        close();
        transport_ = ldms_xprt_new_with_auth(xprt_.c_str(), auth_.c_str(), nullptr);
        if(transport_ == nullptr)
        {
            std::cerr << "Could not create LDMS transport\n";
            return false;
        }
        const int rc = ldms_xprt_connect_by_name(transport_, host_.c_str(), port_.c_str(), nullptr, nullptr);
        if(rc!=0){
            std::cerr << "Could not connect to LDMS at " << host_ << ':' << port_<< " (" << rc << ")\n";
            close();
            return false;
        }
        return true;
    }
    Publisher::~Publisher(){
        close();
    }
}
