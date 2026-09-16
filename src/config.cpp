#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>


/*
Configuration file uses INI format:
    plain-text key-value pairs organized into sections.

Valid sections:
    [tool], [ldms], [pc], [device], [trace]

Single-line comments may begin with '#' or ';'.
*/

namespace rocm_ldms{
    namespace{
        std::string trim(std::string value)
        {
            auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
            value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
            return value;
        }

        bool parse_bool(const std::string& value)
        {
            std::string lower = value;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });

            if(lower == "true" || lower == "yes" || lower == "on" || lower == "1") return true;
            if(lower == "false" || lower == "no" || lower == "off" || lower == "0") return false;
            throw std::runtime_error{"Invalid boolean value: " + value+"."};
        }

        size_t parse_size(const std::string& value, const std::string& key)
        {
            size_t index = 0;
            const unsigned long long parsed = std::stoull(value, &index, 10);
            if(index != value.size() || parsed == 0) throw std::runtime_error{"Invalid " + key + ": " + value+"."};
            return static_cast<size_t>(parsed);
        }

        std::vector<std::string> parse_list(const std::string& value)
        {
            std::vector<std::string> output;
            std::stringstream input{value};
            std::string item;
            while(std::getline(input, item, ','))
            {
                item = trim(item);
                if(!item.empty()) output.push_back(std::move(item));
            }
            return output;
        }

    }
    ToolConfig load_config(const std::string& path){
        std::ifstream input{path};
    if(!input) throw std::runtime_error{"Could not open config file: " + path+"."};

    ToolConfig config{};
    std::string section;
    std::string line;
    size_t line_number = 0;

    while(std::getline(input, line))
    {
        ++line_number;
        line = trim(line);
        if(line.empty() || line.front() == '#' || line.front() == ';') continue;

        if(line.front() == '[' && line.back() == ']')
        {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }

        const size_t equals = line.find('=');
        if(equals == std::string::npos){
            throw std::runtime_error{"Invalid config line " + std::to_string(line_number) + "."};
        }
        const std::string key = trim(line.substr(0,equals));
        const std::string value = trim(line.substr(equals+1));
        const std::string full = section + "." + key;

        if(full == "tool.control_only") config.tool.control_only = parse_bool(value);
        else if(full == "tool.output") config.tool.output = value;
        else if(full == "device.agents") config.device.agents = value;
        else if(full == "device.flush_on_window") config.device.flush_on_window = parse_bool(value);
        else if(full == "tool.window_ms") config.tool.window_ms = parse_size(value, full);
        else if(full == "tool.max_records_per_message")
            config.tool.max_records_per_message = parse_size(value, full);
        else if(full == "ldms.host") config.ldms.host = value;
        else if(full == "ldms.port") config.ldms.port = value;
        else if(full == "ldms.xprt") config.ldms.xprt = value;
        else if(full == "ldms.auth") config.ldms.auth = value;
        else if(full == "pc.enabled") config.pc.enabled = parse_bool(value);
        else if(full == "pc.interval_cycles") config.pc.interval_cycles = parse_size(value, full);
        else if(full == "device.enabled") config.device.enabled = parse_bool(value);
        else if(full == "device.interval_ms") config.device.interval_ms = parse_size(value, full);
        else if(full == "device.counters") config.device.counters = parse_list(value);
        else if(full == "trace.enabled") config.trace.enabled = parse_bool(value);
        else if(full == "trace.hip_runtime") config.trace.hip_runtime = parse_bool(value);
        else if(full == "trace.kernel_dispatch") config.trace.kernel_dispatch = parse_bool(value);
        else if(full == "trace.memory_copy") config.trace.memory_copy = parse_bool(value);
        else if(full == "trace.memory_allocation") config.trace.memory_allocation = parse_bool(value);
        else if(full == "trace.scratch_memory") config.trace.scratch_memory = parse_bool(value);
        else if(full == "trace.runtime_initialization")
            config.trace.runtime_initialization = parse_bool(value);
        else if(full == "trace.correlation_retirement")
            config.trace.correlation_retirement = parse_bool(value);
        else
            throw std::runtime_error{"Unknown config key '" + full + "' on line " +
                                     std::to_string(line_number)+"."};
    }
    if(config.pc.enabled && config.device.enabled) throw std::runtime_error{ "PC.enabled and Device.enabled cannot both be true. PC sampling and device counter collection must be tested as separate modes."};

    if(config.device.enabled && config.device.counters.empty()) throw std::runtime_error{"Device.enabled=true requires at least one device counter."};
    if(config.tool.output != "ldms" && config.tool.output != "serialize" && config.tool.output != "discard")
        throw std::runtime_error{"tool.output must be ldms, serialize, or discard"};
    if(config.tool.output == "discard" && (config.pc.enabled || config.trace.enabled))
        throw std::runtime_error{"discard output is only implemented for device collection"};
    if(config.tool.control_only && (config.pc.enabled || config.device.enabled || config.trace.enabled))
        throw std::runtime_error{"control_only requires all collection modes disabled"};

    return config;
}
}
