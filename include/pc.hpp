#pragma once

#include "config.hpp"
#include "publisher.hpp"
#include "window.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocm_ldms
{


struct PcRecord
{
    uint64_t agent_id = 0;
    uint64_t timestamp = 0;
    uint64_t dispatch_id = 0;

    uint64_t correlation_internal = 0;
    uint64_t correlation_external = 0;

    uint64_t code_object_id = 0;
    uint64_t code_object_offset = 0;

    // Bitmask of active lanes in the sampled wave.
    uint64_t exec_mask = 0;

    uint64_t workgroup_x = 0;
    uint64_t workgroup_y = 0;
    uint64_t workgroup_z = 0;

    // Wave index within the workgroup.
    uint64_t wave_in_group = 0;

    // Number of waves observed by the stochastic sampler.
    uint64_t wave_count = 0;

    // Whether the sampled wave issued an instruction at the sample instant.
    uint64_t wave_issued = 0;

    // ROCProfiler instruction category when the wave issued.
    int64_t instruction_type = -1;

    // ROCProfiler reason category when the wave did not issue.
    int64_t reason_not_issued = -1;

    int64_t issue_valu = -1;
    int64_t issue_matrix = -1;
    int64_t issue_lds = -1;
    int64_t issue_lds_direct = -1;
    int64_t issue_scalar = -1;
    int64_t issue_vmem_tex = -1;
    int64_t issue_flat = -1;
    int64_t issue_exp = -1;
    int64_t issue_misc = -1;
    int64_t issue_brmsg = -1;

    int64_t stall_valu = -1;
    int64_t stall_matrix = -1;
    int64_t stall_lds = -1;
    int64_t stall_lds_direct = -1;
    int64_t stall_scalar = -1;
    int64_t stall_vmem_tex = -1;
    int64_t stall_flat = -1;
    int64_t stall_exp = -1;
    int64_t stall_misc = -1;
    int64_t stall_brmsg = -1;

    int64_t dual_issue_valu = -1;

    // Hardware location information reported with the sample.
    int64_t chiplet = -1;
    int64_t shader_engine = -1;
    int64_t shader_array = -1;
    int64_t cu_or_wgp_id = -1;
    int64_t simd_id = -1;
    int64_t wave_slot = -1;
    int64_t pipe_id = -1;
    int64_t hardware_workgroup_slot = -1;
    int64_t vm_id = -1;
    int64_t queue_id = -1;
    int64_t ace_id = -1;

    // Indicates whether the optional memory-counter fields are valid.
    uint64_t has_memory_counter = 0;

    int64_t memory_load_count = -1;
    int64_t memory_store_count = -1;
    int64_t memory_bvh_count = -1;
    int64_t memory_sample_count = -1;
    int64_t memory_ds_count = -1;
    int64_t memory_km_count = -1;
};

class PcSampler
{
public:
    bool configure(const ToolConfig::Pc& config);

    bool start();
    void stop();
    void flush();
    void shutdown();

    bool publish_window(
        Publisher& publisher,
        const std::string& tag,
        const std::string& producer,
        const std::string& job_id,
        uint64_t pid,
        size_t max_records);

private:
    /*
     * Per-GPU state used by PcSampler.
     *
     * PC sampling is configured per ROCProfiler GPU agent, so each GPU
     * needs its own agent ID, supported configuration list, buffer, and
     * callback thread.
     *
     * PcSampler owns all AgentState objects. Records produced by every
     * configured agent are collected into the shared Window<PcRecord>,
     * while PcRecord::agent_id preserves which GPU produced each sample.
     */
    struct AgentState
    {
        /*
         * ROCProfiler buffer callbacks receive a void* user_data pointer.
         * We pass an AgentState* as that user_data so the callback knows
         * which GPU produced the record. owner then lets the static
         * callback route the sample back to this PcSampler instance.
         */
        PcSampler* owner = nullptr;

        // ROCProfiler identifier for this specific GPU agent.
        rocprofiler_agent_id_t id{};

        // Human-readable agent name, primarily used for logs/debugging.
        std::string name{};

        /*
         * PC-sampling configurations supported by this GPU agent.
         *
         * These are queried from ROCProfiler and used to select the
         * stochastic + cycles configuration supported by the MI300.
         */
        std::vector<rocprofiler_pc_sampling_configuration_t> configurations{};

        // Buffer receiving PC-sampling records for this GPU.
        rocprofiler_buffer_id_t buffer{};

        // Callback thread assigned to process this GPU's buffer.
        rocprofiler_callback_thread_t callback_thread{};

        // Allows cleanup to only flush/destroy successfully created buffers.
        bool has_buffer = false;
    };


    /*
     * ROCProfiler callback used when querying the PC-sampling
     * configurations available for one GPU agent.
     */
    static rocprofiler_status_t collect_configurations(
        const rocprofiler_pc_sampling_configuration_t* configurations,
        size_t count,
        void* user_data);

    // Query and store the PC-sampling configurations available for an agent.
    bool query_configurations(AgentState& agent);

    /*
     * ROCProfiler callback used while discovering available GPU agents.
     * Creates one AgentState for each GPU that supports PC sampling.
     */
    static rocprofiler_status_t collect_gpu_agents(
        rocprofiler_agent_version_t version,
        const void** agents,
        size_t count,
        void* user_data);

    /*
     * Find the stochastic + cycles PC-sampling configuration supported
     * by this GPU agent.
     */
    const rocprofiler_pc_sampling_configuration_t*
    stochastic_cycles_configuration(const AgentState& agent) const;

    // Configure stochastic PC sampling for one specific GPU agent.
    bool configure_agent(AgentState& agent);

    /*
     * Buffer callback invoked by ROCProfiler when PC-sampling records
     * are available for an agent.
     */
    static void buffer_callback(
        rocprofiler_context_id_t context_id,
        rocprofiler_buffer_id_t buffer_id,
        rocprofiler_record_header_t** headers,
        size_t count,
        void* user_data,
        uint64_t drop_count);

    /*
     * Copy one stochastic ROCProfiler sample into our raw PcRecord
     * representation and append it to the active PC window.
     */
    void handle_record(
        AgentState& agent,
        const rocprofiler_pc_sampling_record_stochastic_v0_t& sample);


    // One state object for every GPU agent managed by this sampler.
    std::vector<std::unique_ptr<AgentState>> agents_{};

    // One PC-sampling context managing the configured GPU agents.
    rocprofiler_context_id_t context_{};

    /*
     * Raw PC records from all configured agents are batched here.
     * PcRecord::agent_id identifies which GPU produced each record.
     */
    Window<PcRecord> window_{};

    // Requested stochastic PC-sampling interval in GPU cycles.
    size_t interval_cycles_ = 0;

    // Lifecycle state used to safely start/stop valid contexts.
    bool context_created_ = false;
    bool running_ = false;
};

}  // namespace rocm_ldms