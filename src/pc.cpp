#include "pc.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

namespace rocm_ldms
{
namespace
{

const size_t kBufferSize = 8 * 1024 * 1024;  // 8 MB ROCprofiler delivery buffer
const size_t kWatermark = 4 * 1024 * 1024;   // Threshold for delivery

// Check the status of ROCprofiler calls with this function to reduce
// repeated status checks throughout the code.
bool check_status(rocprofiler_status_t status, const char* operation)
{
    if(status == ROCPROFILER_STATUS_SUCCESS)
        return true;

    std::cerr
        << "rocprof_ldms: "
        << operation
        << " failed with ROCProfiler status "
        << status
        << '\n';

    return false;
}

// Stochastic PC sampling can require a power-of-two cycle interval.
size_t nearest_supported_power_of_two(
    size_t value,
    size_t minimum,
    size_t maximum)
{
    size_t power = 1;

    while(power < value &&
          power <= std::numeric_limits<size_t>::max() / 2)
    {
        power <<= 1;
    }

    if(power > maximum)
    {
        power = 1;

        while(power <= maximum / 2)
            power <<= 1;
    }

    while(power < minimum &&
          power <= std::numeric_limits<size_t>::max() / 2)
    {
        power <<= 1;
    }

    return std::clamp(power, minimum, maximum);
}

// Escape string values before placing them inside the JSON LDMS message.
std::string json_escape(const std::string& value)
{
    std::string out;

    for(char ch : value)
    {
        if(ch == '"')
            out += "\\\"";
        else if(ch == '\\')
            out += "\\\\";
        else if(ch == '\n')
            out += "\\n";
        else if(ch == '\r')
            out += "\\r";
        else if(ch == '\t')
            out += "\\t";
        else
            out += ch;
    }

    return out;
}

// Serialize one internal PcRecord into the JSON records array.
void write_pc_record(
    std::ostringstream& out,
    const PcRecord& r)
{
    out << '{'
        // OVIS JSON integers are signed 64-bit. Keep all unsigned values as
        // decimal strings so values above INT64_MAX survive transport exactly.
        << "\"agent_id\":\"" << r.agent_id << "\"," 
        << "\"timestamp\":\"" << r.timestamp << "\"," 
        << "\"dispatch_id\":\"" << r.dispatch_id << "\"," 
        << "\"correlation_internal\":\"" << r.correlation_internal << "\"," 
        << "\"correlation_external\":\"" << r.correlation_external << "\"," 
        << "\"code_object_id\":\"" << r.code_object_id << "\"," 
        << "\"code_object_offset\":\"" << r.code_object_offset << "\"," 
        << "\"exec_mask\":\"" << r.exec_mask << "\"," 
        << "\"workgroup_x\":\"" << r.workgroup_x << "\"," 
        << "\"workgroup_y\":\"" << r.workgroup_y << "\"," 
        << "\"workgroup_z\":\"" << r.workgroup_z << "\"," 
        << "\"wave_in_group\":\"" << r.wave_in_group << "\"," 
        << "\"wave_count\":\"" << r.wave_count << "\"," 
        << "\"wave_issued\":\"" << r.wave_issued << "\"," 
        << "\"instruction_type\":" << r.instruction_type << ','
        << "\"reason_not_issued\":" << r.reason_not_issued << ','
        << "\"issue_valu\":" << r.issue_valu << ','
        << "\"issue_matrix\":" << r.issue_matrix << ','
        << "\"issue_lds\":" << r.issue_lds << ','
        << "\"issue_lds_direct\":" << r.issue_lds_direct << ','
        << "\"issue_scalar\":" << r.issue_scalar << ','
        << "\"issue_vmem_tex\":" << r.issue_vmem_tex << ','
        << "\"issue_flat\":" << r.issue_flat << ','
        << "\"issue_exp\":" << r.issue_exp << ','
        << "\"issue_misc\":" << r.issue_misc << ','
        << "\"issue_brmsg\":" << r.issue_brmsg << ','
        << "\"stall_valu\":" << r.stall_valu << ','
        << "\"stall_matrix\":" << r.stall_matrix << ','
        << "\"stall_lds\":" << r.stall_lds << ','
        << "\"stall_lds_direct\":" << r.stall_lds_direct << ','
        << "\"stall_scalar\":" << r.stall_scalar << ','
        << "\"stall_vmem_tex\":" << r.stall_vmem_tex << ','
        << "\"stall_flat\":" << r.stall_flat << ','
        << "\"stall_exp\":" << r.stall_exp << ','
        << "\"stall_misc\":" << r.stall_misc << ','
        << "\"stall_brmsg\":" << r.stall_brmsg << ','
        << "\"dual_issue_valu\":" << r.dual_issue_valu << ','
        << "\"chiplet\":" << r.chiplet << ','
        << "\"shader_engine\":" << r.shader_engine << ','
        << "\"shader_array\":" << r.shader_array << ','
        << "\"cu_or_wgp_id\":" << r.cu_or_wgp_id << ','
        << "\"simd_id\":" << r.simd_id << ','
        << "\"wave_slot\":" << r.wave_slot << ','
        << "\"pipe_id\":" << r.pipe_id << ','
        << "\"hardware_workgroup_slot\":" << r.hardware_workgroup_slot << ','
        << "\"vm_id\":" << r.vm_id << ','
        << "\"queue_id\":" << r.queue_id << ','
        << "\"ace_id\":" << r.ace_id << ','
        << "\"has_memory_counter\":\"" << r.has_memory_counter << "\"," 
        << "\"memory_load_count\":" << r.memory_load_count << ','
        << "\"memory_store_count\":" << r.memory_store_count << ','
        << "\"memory_bvh_count\":" << r.memory_bvh_count << ','
        << "\"memory_sample_count\":" << r.memory_sample_count << ','
        << "\"memory_ds_count\":" << r.memory_ds_count << ','
        << "\"memory_km_count\":" << r.memory_km_count
        << '}';
}

}  // namespace


// Callback for rocprofiler_query_pc_sampling_agent_configurations(...).
// Copies the configurations returned by ROCprofiler into our AgentState vector.
rocprofiler_status_t PcSampler::collect_configurations(
    const rocprofiler_pc_sampling_configuration_t* configurations,
    size_t count,
    void* user_data)
{
    auto* output =
        static_cast<
            std::vector<rocprofiler_pc_sampling_configuration_t>*>(
                user_data);

    output->clear();

    if(count != 0)
    {
        output->assign(
            configurations,
            configurations + count);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}


// Lists what PC-sampling configurations a GPU agent currently supports.
bool PcSampler::query_configurations(
    AgentState& agent)
{
    agent.configurations.clear();

    const auto status =
        rocprofiler_query_pc_sampling_agent_configurations(
            agent.id,
            collect_configurations,
            &agent.configurations);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr
            << "rocprof_ldms: PC sampling unavailable for GPU '"
            << agent.name
            << "'\n";

        return false;
    }

    return !agent.configurations.empty();
}


// Callback function for rocprofiler_query_available_agents(...).
// Discovers GPU agents, creates an AgentState for each GPU,
// queries its supported PC-sampling configurations,
// and stores only GPUs that support PC sampling.
rocprofiler_status_t PcSampler::collect_gpu_agents(
    rocprofiler_agent_version_t version,
    const void** agents,
    size_t count,
    void* user_data)
{
    if(version != ROCPROFILER_AGENT_INFO_VERSION_0 ||
       user_data == nullptr)
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    // Recover the PcSampler instance passed into
    // rocprofiler_query_available_agents(...).
    auto* self =
        static_cast<PcSampler*>(user_data);

    // We requested AGENT_INFO_VERSION_0, so the returned agents
    // can be interpreted as rocprofiler_agent_t objects.
    auto* agent_list =
        reinterpret_cast<const rocprofiler_agent_t**>(agents);

    for(size_t i = 0; i < count; ++i)
    {
        // PC sampling is only needed for GPU agents.
        if(agent_list[i]->type !=
           ROCPROFILER_AGENT_TYPE_GPU)
        {
            continue;
        }

        // Maintain our own per-GPU state containing the ROCprofiler
        // agent ID, buffer, callback thread, and PC configurations.
        auto agent =
            std::make_unique<AgentState>();

        agent->owner = self;

        agent->id =
            agent_list[i]->id;

        agent->name =
            agent_list[i]->name != nullptr
                ? agent_list[i]->name
                : "unknown";

        // Only store GPU agents for which PC sampling is available.
        if(self->query_configurations(*agent))
        {
            self->agents_.emplace_back(
                std::move(agent));
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}


// Find the stochastic/cycles PC-sampling configuration that this
// sampler uses on supported GPUs.
const rocprofiler_pc_sampling_configuration_t*
PcSampler::stochastic_cycles_configuration(
    const AgentState& agent) const
{
    const auto itr =
        std::find_if(
            agent.configurations.begin(),
            agent.configurations.end(),
            [](const rocprofiler_pc_sampling_configuration_t& configuration)
            {
                return
                    configuration.method ==
                        ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC
                    &&
                    configuration.unit ==
                        ROCPROFILER_PC_SAMPLING_UNIT_CYCLES;
            });

    return
        itr == agent.configurations.end()
            ? nullptr
            : &*itr;
}


// Configure stochastic/cycles PC sampling for one GPU agent.
bool PcSampler::configure_agent(
    AgentState& agent)
{
    // Re-query immediately before configuration because PC-sampling
    // availability can change between discovery and configuration.
    if(!query_configurations(agent))
        return false;

    const auto* configuration =
        stochastic_cycles_configuration(agent);

    if(configuration == nullptr)
    {
        std::cerr
            << "rocprof_ldms: GPU '"
            << agent.name
            << "' has no stochastic/cycles "
               "PC-sampling configuration\n";

        return false;
    }

    // Keep the configured interval within the bounds reported
    // by ROCprofiler for this GPU.
    size_t interval =
        std::clamp(
            interval_cycles_,
            configuration->min_interval,
            configuration->max_interval);

    // Some stochastic configurations require a power-of-two interval.
    if((configuration->flags &
        ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2)
       != 0)
    {
        interval =
            nearest_supported_power_of_two(
                interval,
                configuration->min_interval,
                configuration->max_interval);
    }

    // Associate stochastic PC sampling on this GPU with this context
    // and this GPU's ROCprofiler buffer.
    const auto status =
        rocprofiler_configure_pc_sampling_service(
            context_,
            agent.id,
            ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
            ROCPROFILER_PC_SAMPLING_UNIT_CYCLES,
            interval,
            agent.buffer,
            0);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        return check_status(
            status,
            "rocprofiler_configure_pc_sampling_service");
    }

    std::cerr
        << "rocprof_ldms: stochastic PC sampling GPU '"
        << agent.name
        << "' every "
        << interval
        << " cycles\n";

    return true;
}


// Configure the complete ROCprofiler PC-sampling service.
// Creates the context, discovers usable GPUs, creates one buffer
// per GPU, configures PC sampling, and assigns callback threads.
bool PcSampler::configure(
    const ToolConfig::Pc& config)
{
    interval_cycles_ =
        config.interval_cycles;  // Save the configuration.

    window_.reset();  // Clear the LDMS window.

    // Create the ROCprofiler context that owns the PC-sampling services.
    if(!check_status(
           rocprofiler_create_context(
               &context_),
           "rocprofiler_create_context (pc)"))
    {
        return false;
    }

    context_created_ = true;

    // Discover available agents. ROCprofiler invokes collect_gpu_agents(),
    // which filters GPUs and queries their PC-sampling configurations.
    if(!check_status(
           rocprofiler_query_available_agents(
               ROCPROFILER_AGENT_INFO_VERSION_0,
               collect_gpu_agents,
               sizeof(rocprofiler_agent_t),
               this),
           "rocprofiler_query_available_agents (pc)"))
    {
        return false;
    }

    if(agents_.empty())
    {
        std::cerr
            << "rocprof_ldms: no GPU with stochastic "
               "PC sampling was found\n";

        return false;
    }

    for(auto& agent : agents_)
    {
        // Create one ROCprofiler delivery buffer per GPU.
        // PC-sampling records generated for this GPU are delivered
        // through buffer_callback().
        if(!check_status(
               rocprofiler_create_buffer(
                   context_,
                   kBufferSize,
                   kWatermark,
                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                   buffer_callback,
                   agent.get(),
                   &agent->buffer),
               "rocprofiler_create_buffer (pc)"))
        {
            return false;
        }

        agent->has_buffer = true;

        // Configure stochastic/cycles PC sampling for this GPU and
        // associate the sampling service with its buffer.
        if(!configure_agent(*agent))
            return false;

        // Create and assign a dedicated ROCprofiler callback thread
        // for this GPU's buffer so buffered PC-sampling records are
        // delivered independently of other buffers.
        if(!check_status(
               rocprofiler_create_callback_thread(
                   &agent->callback_thread),
               "rocprofiler_create_callback_thread (pc)")
           ||
           !check_status(
               rocprofiler_assign_callback_thread(
                   agent->buffer,
                   agent->callback_thread),
               "rocprofiler_assign_callback_thread (pc)"))
        {
            return false;
        }
    }

    // Verify that the completed ROCprofiler context is valid before
    // attempting to start PC sampling.
    int valid = 0;

    return
        check_status(
            rocprofiler_context_is_valid(
                context_,
                &valid),
            "rocprofiler_context_is_valid (pc)")
        &&
        valid != 0;
}


// Activate the configured PC-sampling context.
bool PcSampler::start()
{
    if(!context_created_ || running_)
        return context_created_;

    if(!check_status(
           rocprofiler_start_context(
               context_),
           "rocprofiler_start_context (pc)"))
    {
        return false;
    }

    running_ = true;

    return true;
}


// Stop collection without destroying the configured context/buffers.
void PcSampler::stop()
{
    if(!running_)
        return;

    int active = 0;
    const rocprofiler_status_t query_status =
        rocprofiler_context_is_active(context_, &active);

    if(query_status == ROCPROFILER_STATUS_SUCCESS && active != 0)
    {
        check_status(
            rocprofiler_stop_context(context_),
            "rocprofiler_stop_context (pc)");
    }
    else if(query_status != ROCPROFILER_STATUS_SUCCESS &&
            query_status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
    {
        check_status(
            query_status,
            "rocprofiler_context_is_active (pc)");
    }

    running_ = false;
}


// Force any records still waiting in each ROCprofiler buffer
// through buffer_callback().
void PcSampler::flush()
{
    for(auto& agent : agents_)
    {
        if(agent->has_buffer)
        {
            check_status(
                rocprofiler_flush_buffer(
                    agent->buffer),
                "rocprofiler_flush_buffer (pc)");
        }
    }
}


// Stop PC sampling, flush pending samples, destroy ROCprofiler
// buffers, and release our per-GPU AgentState objects.
void PcSampler::shutdown()
{
    stop();

    flush();

    for(auto& agent : agents_)
    {
        if(!agent->has_buffer)
            continue;

        check_status(
            rocprofiler_destroy_buffer(
                agent->buffer),
            "rocprofiler_destroy_buffer (pc)");

        agent->has_buffer = false;
    }

    agents_.clear();
}


// Convert one ROCprofiler stochastic-v0 PC-sampling record into
// our internal PcRecord and add it to the current LDMS window.
void PcSampler::handle_record(
    AgentState& agent,
    const rocprofiler_pc_sampling_record_stochastic_v0_t& sample)
{
    PcRecord record{};

    record.agent_id =
        agent.id.handle;

    record.timestamp =
        sample.timestamp;

    record.dispatch_id =
        sample.dispatch_id;

    record.correlation_internal =
        sample.correlation_id.internal;

    record.correlation_external =
        sample.correlation_id.external.value;

    record.code_object_id =
        sample.pc.code_object_id;

    record.code_object_offset =
        sample.pc.code_object_offset;

    record.exec_mask =
        sample.exec_mask;

    record.workgroup_x =
        sample.workgroup_id.x;

    record.workgroup_y =
        sample.workgroup_id.y;

    record.workgroup_z =
        sample.workgroup_id.z;

    record.wave_in_group =
        sample.wave_in_group;

    record.wave_count =
        sample.wave_count;

    record.wave_issued =
        sample.wave_issued;

    record.instruction_type =
        sample.inst_type;


    // Scheduler / issue-state snapshot.
    record.reason_not_issued =
        sample.snapshot.reason_not_issued;

    record.issue_valu =
        sample.snapshot.arb_state_issue_valu;

    record.issue_matrix =
        sample.snapshot.arb_state_issue_matrix;

    record.issue_lds =
        sample.snapshot.arb_state_issue_lds;

    record.issue_lds_direct =
        sample.snapshot.arb_state_issue_lds_direct;

    record.issue_scalar =
        sample.snapshot.arb_state_issue_scalar;

    record.issue_vmem_tex =
        sample.snapshot.arb_state_issue_vmem_tex;

    record.issue_flat =
        sample.snapshot.arb_state_issue_flat;

    record.issue_exp =
        sample.snapshot.arb_state_issue_exp;

    record.issue_misc =
        sample.snapshot.arb_state_issue_misc;

    record.issue_brmsg =
        sample.snapshot.arb_state_issue_brmsg;


    // Scheduler stall-state snapshot.
    record.stall_valu =
        sample.snapshot.arb_state_stall_valu;

    record.stall_matrix =
        sample.snapshot.arb_state_stall_matrix;

    record.stall_lds =
        sample.snapshot.arb_state_stall_lds;

    record.stall_lds_direct =
        sample.snapshot.arb_state_stall_lds_direct;

    record.stall_scalar =
        sample.snapshot.arb_state_stall_scalar;

    record.stall_vmem_tex =
        sample.snapshot.arb_state_stall_vmem_tex;

    record.stall_flat =
        sample.snapshot.arb_state_stall_flat;

    record.stall_exp =
        sample.snapshot.arb_state_stall_exp;

    record.stall_misc =
        sample.snapshot.arb_state_stall_misc;

    record.stall_brmsg =
        sample.snapshot.arb_state_stall_brmsg;

    record.dual_issue_valu =
        sample.snapshot.dual_issue_valu;


    // Physical/logical GPU hardware location of the sampled wave.
    record.chiplet =
        sample.hw_id.chiplet;

    record.shader_engine =
        sample.hw_id.shader_engine_id;

    record.shader_array =
        sample.hw_id.shader_array_id;

    record.cu_or_wgp_id =
        sample.hw_id.cu_or_wgp_id;

    record.simd_id =
        sample.hw_id.simd_id;

    record.wave_slot =
        sample.hw_id.wave_id;

    record.pipe_id =
        sample.hw_id.pipe_id;

    record.hardware_workgroup_slot =
        sample.hw_id.workgroup_id;

    record.vm_id =
        sample.hw_id.vm_id;

    record.queue_id =
        sample.hw_id.queue_id;

    record.ace_id =
        sample.hw_id.microengine_id;


    // Memory counters are only meaningful when ROCprofiler marks
    // them as present for this stochastic sample.
    record.has_memory_counter =
        sample.flags.has_memory_counter;

    if(sample.flags.has_memory_counter)
    {
        record.memory_load_count =
            sample.memory_counters.load_cnt;

        record.memory_store_count =
            sample.memory_counters.store_cnt;

        record.memory_bvh_count =
            sample.memory_counters.bvh_cnt;

        record.memory_sample_count =
            sample.memory_counters.sample_cnt;

        record.memory_ds_count =
            sample.memory_counters.ds_cnt;

        record.memory_km_count =
            sample.memory_counters.km_cnt;
    }

    // ROCprofiler's job ends here. From this point on the sample is
    // represented by our own PcRecord and managed by our LDMS window.
    window_.add(
        std::move(record));
}


// Rotate the current PC-sampling window and publish all collected
// PcRecords as JSON LDMS messages.
bool PcSampler::publish_window(
    Publisher& publisher,
    const std::string& tag,
    const std::string& producer,
    const std::string& job_id,
    uint64_t pid,
    size_t max_records)
{
    auto snapshot =
        window_.rotate();

    if(snapshot.records.empty())
        return true;

    if(max_records == 0)
        return false;

    const size_t chunk_count =
        (snapshot.records.size() + max_records - 1) / max_records;

    for(size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index)
    {
        const size_t begin = chunk_index * max_records;
        const size_t end = std::min(begin + max_records, snapshot.records.size());

        std::ostringstream out;

        // Minimum metadata needed to identify the node/job/process/window,
        // followed by the complete set of PC samples from this window.
        out << '{'
            << "\"schema\":\"rocm_pc_raw_window\","
            << "\"producer\":\"" << json_escape(producer) << "\","
            << "\"job_id\":\"" << json_escape(job_id) << "\","
            << "\"pid\":" << pid << ','
            << "\"window_start_ns\":" << snapshot.start_ns << ','
            << "\"window_end_ns\":" << snapshot.end_ns << ','
            << "\"window_sequence\":" << snapshot.sequence << ','
            << "\"chunk_index\":" << chunk_index << ','
            << "\"chunk_count\":" << chunk_count << ','
            << "\"records_max_len\":" << max_records << ','
            << "\"records\":[";

        for(size_t i = begin; i < end; ++i)
        {
            if(i != begin)
                out << ',';

            write_pc_record(
                out,
                snapshot.records[i]);
        }

        out << "]}";

        // Send the entire logical PC-sampling window as LDMS messages.
        if(!publisher.publish(
               tag,
               out.str()))
        {
            return false;
        }
    }

    return true;
}


// ROCprofiler buffer callback used at runtime.
// Receives batches of generic ROCprofiler records, filters for valid
// stochastic-v0 PC samples, and passes each sample to handle_record().
void PcSampler::buffer_callback(
    rocprofiler_context_id_t,
    rocprofiler_buffer_id_t,
    rocprofiler_record_header_t** headers,
    size_t count,
    void* user_data,
    uint64_t drop_count)
{
    // create_buffer() stored this GPU's AgentState as callback user_data.
    auto* agent =
        static_cast<AgentState*>(
            user_data);

    if(agent == nullptr ||
       agent->owner == nullptr)
    {
        return;
    }

    if(drop_count != 0)
    {
        std::cerr
            << "rocprof_ldms: ROCProfiler PC buffer dropped "
            << drop_count
            << " records\n";
    }

    for(size_t i = 0;
        i < count;
        ++i)
    {
        const auto* header =
            headers[i];

        // Validate the generic ROCprofiler record header before using it.
        if(header == nullptr ||
           header->hash !=
               rocprofiler_record_header_compute_hash(
                   header->category,
                   header->kind))
        {
            continue;
        }

        // Ignore records that are not from the PC-sampling service.
        if(header->category !=
           ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            continue;
        }

        // Ignore explicitly invalid PC-sampling records.
        if(header->kind ==
           ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE)
        {
            continue;
        }

        // This sampler only handles stochastic-v0 PC samples.
        if(header->kind !=
           ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE)
        {
            continue;
        }

        // We verified the record kind, so its payload can now be
        // interpreted as a stochastic-v0 PC-sampling record.
        const auto* sample =
            static_cast<
                const rocprofiler_pc_sampling_record_stochastic_v0_t*>(
                    header->payload);

        if(sample != nullptr)
        {
            agent->owner->handle_record(
                *agent,
                *sample);
        }
    }
}

}  // namespace rocm_ldms
