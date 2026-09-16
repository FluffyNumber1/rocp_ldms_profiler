extern "C" {
#include <ldms/ldms.h>
#include <ldms/ldmsd_plug_api.h>
#include <ovis_json/ovis_json.h>
#include <ovis_log/ovis_log.h>
}

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

struct attr_value_list;

extern "C" {
char *av_value(struct attr_value_list *avl, const char *name);
}

/* Minimal ABI compatible with the LDMS 4.5 sampler plugin interface. */
enum ldmsd_plugin_type {
  LDMSD_PLUGIN_SAMPLER = 1,
  LDMSD_PLUGIN_STORE,
};

struct ldmsd_plugin {
  enum ldmsd_plugin_type type;
  uint64_t flags;
  int (*config)(ldmsd_plug_handle_t, attr_value_list *, attr_value_list *);
  const char *(*usage)(ldmsd_plug_handle_t);
  int (*constructor)(ldmsd_plug_handle_t);
  void (*destructor)(ldmsd_plug_handle_t);
  void *reserved[8];
};

struct ldmsd_sampler {
  ldmsd_plugin base;
  int (*sample)(ldmsd_plug_handle_t);
};

namespace {

constexpr size_t PRODUCER_LEN = 256;
constexpr size_t JOB_ID_LEN = 128;

struct FieldSpec {
  const char *name;
  ldms_value_type type;
  size_t count = 1;
};

/* One explicit table per mode: these tables are the fixed LDMS schemas. */
constexpr std::array<FieldSpec, 55> PC_FIELDS{{
    {"agent_id", LDMS_V_U64},
    {"timestamp", LDMS_V_U64},
    {"dispatch_id", LDMS_V_U64},
    {"correlation_internal", LDMS_V_U64},
    {"correlation_external", LDMS_V_U64},
    {"code_object_id", LDMS_V_U64},
    {"code_object_offset", LDMS_V_U64},
    {"exec_mask", LDMS_V_U64},
    {"workgroup_x", LDMS_V_U64},
    {"workgroup_y", LDMS_V_U64},
    {"workgroup_z", LDMS_V_U64},
    {"wave_in_group", LDMS_V_U64},
    {"wave_count", LDMS_V_U64},
    {"wave_issued", LDMS_V_U64},
    {"instruction_type", LDMS_V_S64},
    {"reason_not_issued", LDMS_V_S64},
    {"issue_valu", LDMS_V_S64},
    {"issue_matrix", LDMS_V_S64},
    {"issue_lds", LDMS_V_S64},
    {"issue_lds_direct", LDMS_V_S64},
    {"issue_scalar", LDMS_V_S64},
    {"issue_vmem_tex", LDMS_V_S64},
    {"issue_flat", LDMS_V_S64},
    {"issue_exp", LDMS_V_S64},
    {"issue_misc", LDMS_V_S64},
    {"issue_brmsg", LDMS_V_S64},
    {"stall_valu", LDMS_V_S64},
    {"stall_matrix", LDMS_V_S64},
    {"stall_lds", LDMS_V_S64},
    {"stall_lds_direct", LDMS_V_S64},
    {"stall_scalar", LDMS_V_S64},
    {"stall_vmem_tex", LDMS_V_S64},
    {"stall_flat", LDMS_V_S64},
    {"stall_exp", LDMS_V_S64},
    {"stall_misc", LDMS_V_S64},
    {"stall_brmsg", LDMS_V_S64},
    {"dual_issue_valu", LDMS_V_S64},
    {"chiplet", LDMS_V_S64},
    {"shader_engine", LDMS_V_S64},
    {"shader_array", LDMS_V_S64},
    {"cu_or_wgp_id", LDMS_V_S64},
    {"simd_id", LDMS_V_S64},
    {"wave_slot", LDMS_V_S64},
    {"pipe_id", LDMS_V_S64},
    {"hardware_workgroup_slot", LDMS_V_S64},
    {"vm_id", LDMS_V_S64},
    {"queue_id", LDMS_V_S64},
    {"ace_id", LDMS_V_S64},
    {"has_memory_counter", LDMS_V_U64},
    {"memory_load_count", LDMS_V_S64},
    {"memory_store_count", LDMS_V_S64},
    {"memory_bvh_count", LDMS_V_S64},
    {"memory_sample_count", LDMS_V_S64},
    {"memory_ds_count", LDMS_V_S64},
    {"memory_km_count", LDMS_V_S64},
}};

constexpr std::array<FieldSpec, 8> DEVICE_FIELDS{{
    {"agent_id", LDMS_V_U64},
    {"sample_timestamp", LDMS_V_U64},
    {"counter_instance_id", LDMS_V_U64},
    {"counter_id", LDMS_V_U64},
    {"dispatch_id", LDMS_V_U64},
    {"counter_name", LDMS_V_CHAR_ARRAY, 128},
    {"value_valid", LDMS_V_U64},
    {"value", LDMS_V_D64},
}};

constexpr std::array<FieldSpec, 31> TRACE_FIELDS{{
    {"valid_fields", LDMS_V_U64},
    {"kind", LDMS_V_U64},
    {"operation", LDMS_V_U64},
    {"kind_name", LDMS_V_CHAR_ARRAY, 64},
    {"operation_name", LDMS_V_CHAR_ARRAY, 128},
    {"start_timestamp", LDMS_V_U64},
    {"end_timestamp", LDMS_V_U64},
    {"correlation_internal", LDMS_V_U64},
    {"correlation_external", LDMS_V_U64},
    {"thread_id", LDMS_V_U64},
    {"agent_id", LDMS_V_U64},
    {"queue_id", LDMS_V_U64},
    {"kernel_id", LDMS_V_U64},
    {"dispatch_id", LDMS_V_U64},
    {"src_agent_id", LDMS_V_U64},
    {"dst_agent_id", LDMS_V_U64},
    {"bytes", LDMS_V_U64},
    {"address", LDMS_V_U64},
    {"src_address", LDMS_V_U64},
    {"dst_address", LDMS_V_U64},
    {"flags", LDMS_V_U64},
    {"version", LDMS_V_U64},
    {"instance", LDMS_V_U64},
    {"private_segment_size", LDMS_V_U64},
    {"group_segment_size", LDMS_V_U64},
    {"workgroup_x", LDMS_V_U64},
    {"workgroup_y", LDMS_V_U64},
    {"workgroup_z", LDMS_V_U64},
    {"grid_x", LDMS_V_U64},
    {"grid_y", LDMS_V_U64},
    {"grid_z", LDMS_V_U64},
}};

struct Layout {
  ldms_record_t record_def = nullptr;
  int record_type_idx = -1;
  std::vector<int> field_idx{};
  int producer = -1;
  int job_id = -1;
  int pid = -1;
  int window_start_ns = -1;
  int window_end_ns = -1;
  int window_sequence = -1;
  int chunk_index = -1;
  int chunk_count = -1;
  int records = -1;
};

struct FieldValue {
  uint64_t u64 = 0;
  int64_t s64 = 0;
  double d64 = 0.0;
  std::string string{};
};

using RecordValues = std::vector<FieldValue>;

struct Bridge {
  Bridge(const char *schema_value, const char *record_value,
         const char *tag_value, const char *instance_value,
         const FieldSpec *field_values, size_t field_count_value)
      : schema_name{schema_value}, record_name{record_value},
        message_tag{tag_value}, instance{instance_value}, fields{field_values},
        field_count{field_count_value} {}

  const char *schema_name;
  const char *record_name;
  std::string message_tag;
  std::string instance;
  const FieldSpec *fields;
  size_t field_count;

  ovis_log_t log = nullptr;
  std::string producer = "rocm_sampler";
  uint32_t max_records = 256;
  ldms_schema_t schema = nullptr;
  ldms_set_t set = nullptr;
  Layout layout{};
  ldms_msg_client_t msg_client = nullptr;
  std::mutex update_mutex{};
};

Bridge g_pc{"rocm_pc_raw_window", "rocm_pc_record",
            "rocm_pc_raw",        "rocm_sampler/rocm_pc_raw_window",
            PC_FIELDS.data(),     PC_FIELDS.size()};
Bridge g_device{"rocm_device_raw_window", "rocm_device_record",
                "rocm_device_raw",        "rocm_sampler/rocm_device_raw_window",
                DEVICE_FIELDS.data(),     DEVICE_FIELDS.size()};
Bridge g_trace{"rocm_trace_raw_window", "rocm_trace_record",
               "rocm_trace_raw",        "rocm_sampler/rocm_trace_raw_window",
               TRACE_FIELDS.data(),     TRACE_FIELDS.size()};

const char *usage(ldmsd_plug_handle_t) {
  return "config name=rocprof_msg_sampler producer=<producer> "
         "[pc_instance=<name>] [device_instance=<name>] "
         "[trace_instance=<name>] [max_records=256]\n";
}

bool add_metric(ldms_schema_t schema, int &idx, const char *name,
                ldms_value_type type, size_t count = 1) {
  if (type == LDMS_V_CHAR_ARRAY)
    idx = ldms_schema_metric_array_add(schema, name, type, count);
  else
    idx = ldms_schema_metric_add(schema, name, type);
  if (idx < 0)
    std::cerr << "rocprof_msg_sampler: failed to add metric " << name << '\n';
  return idx >= 0;
}

bool create_schema_and_set(Bridge &bridge) {
  auto &layout = bridge.layout;
  bridge.schema = ldms_schema_new(bridge.schema_name);
  if (!bridge.schema)
    return false;

  if (!add_metric(bridge.schema, layout.producer, "producer", LDMS_V_CHAR_ARRAY,
                  PRODUCER_LEN) ||
      !add_metric(bridge.schema, layout.job_id, "job_id", LDMS_V_CHAR_ARRAY,
                  JOB_ID_LEN) ||
      !add_metric(bridge.schema, layout.pid, "pid", LDMS_V_U64) ||
      !add_metric(bridge.schema, layout.window_start_ns, "window_start_ns",
                  LDMS_V_U64) ||
      !add_metric(bridge.schema, layout.window_end_ns, "window_end_ns",
                  LDMS_V_U64) ||
      !add_metric(bridge.schema, layout.window_sequence, "window_sequence",
                  LDMS_V_U64) ||
      !add_metric(bridge.schema, layout.chunk_index, "chunk_index",
                  LDMS_V_U64) ||
      !add_metric(bridge.schema, layout.chunk_count, "chunk_count", LDMS_V_U64))
    return false;

  layout.record_def = ldms_record_create(bridge.record_name);
  if (!layout.record_def)
    return false;

  layout.field_idx.resize(bridge.field_count);
  for (size_t i = 0; i < bridge.field_count; ++i) {
    const auto &field = bridge.fields[i];
    layout.field_idx[i] = ldms_record_metric_add(
        layout.record_def, field.name, nullptr, field.type, field.count);
    if (layout.field_idx[i] < 0) {
      std::cerr << "rocprof_msg_sampler: failed to add record field "
                << field.name << '\n';
      return false;
    }
  }

  layout.record_type_idx =
      ldms_schema_record_add(bridge.schema, layout.record_def);
  if (layout.record_type_idx < 0)
    return false;

  const size_t heap_size = static_cast<size_t>(bridge.max_records) *
                           ldms_record_heap_size_get(layout.record_def);
  layout.records =
      ldms_schema_metric_list_add(bridge.schema, "records", nullptr, heap_size);
  if (layout.records < 0)
    return false;

  bridge.set = ldms_set_new(bridge.instance.c_str(), bridge.schema);
  if (!bridge.set)
    return false;
  ldms_set_producer_name_set(bridge.set, bridge.producer.c_str());
  ldms_set_data_copy_set(bridge.set, 1);
  return ldms_set_publish(bridge.set) == 0;
}

json_entity_t required_value(json_entity_t object, const char *name,
                             enum json_value_e type) {
  json_entity_t value = json_value_find(object, name);
  if (!value || json_entity_type(value) != type) {
    std::cerr << "rocprof_msg_sampler: missing or invalid JSON field " << name
              << '\n';
    return nullptr;
  }
  return value;
}

bool read_u64_entity(json_entity_t value, uint64_t &output) {
  if (json_entity_type(value) == JSON_INT_VALUE) {
    const int64_t parsed = json_value_int(value);
    if (parsed < 0)
      return false;
    output = static_cast<uint64_t>(parsed);
    return true;
  }
  if (json_entity_type(value) != JSON_STRING_VALUE)
    return false;

  const char *text = json_value_cstr(value);
  if (!text || !*text)
    return false;
  for (const char *digit = text; *digit; ++digit) {
    if (*digit < '0' || *digit > '9')
      return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0')
    return false;
  output = static_cast<uint64_t>(parsed);
  return true;
}

bool read_u64(json_entity_t object, const char *name, uint64_t &output) {
  json_entity_t value = json_value_find(object, name);
  return value && read_u64_entity(value, output);
}

bool read_field(json_entity_t object, const FieldSpec &field,
                FieldValue &output) {
  json_entity_t value = json_value_find(object, field.name);
  if (!value)
    return false;

  switch (field.type) {
  case LDMS_V_U64:
    return read_u64_entity(value, output.u64);
  case LDMS_V_S64:
    if (json_entity_type(value) != JSON_INT_VALUE)
      return false;
    output.s64 = json_value_int(value);
    return true;
  case LDMS_V_D64:
    if (json_entity_type(value) == JSON_FLOAT_VALUE)
      output.d64 = json_value_float(value);
    else if (json_entity_type(value) == JSON_INT_VALUE)
      output.d64 = static_cast<double>(json_value_int(value));
    else
      return false;
    return true;
  case LDMS_V_CHAR_ARRAY:
    if (json_entity_type(value) != JSON_STRING_VALUE)
      return false;
    output.string = json_value_cstr(value);
    return true;
  default:
    return false;
  }
}

bool parse_records(Bridge &bridge, json_entity_t message,
                   std::vector<RecordValues> &output) {
  json_entity_t records = required_value(message, "records", JSON_LIST_VALUE);
  if (!records)
    return false;

  output.clear();
  output.reserve(bridge.max_records);
  for (json_entity_t item = json_item_first(records); item;
       item = json_item_next(item)) {
    if (output.size() == bridge.max_records) {
      std::cerr << "rocprof_msg_sampler: record batch exceeds max_records; "
                   "truncating\n";
      break;
    }
    if (json_entity_type(item) != JSON_DICT_VALUE)
      return false;

    RecordValues values(bridge.field_count);
    for (size_t i = 0; i < bridge.field_count; ++i) {
      if (!read_field(item, bridge.fields[i], values[i])) {
        std::cerr << "rocprof_msg_sampler: invalid JSON field "
                  << bridge.fields[i].name << '\n';
        return false;
      }
    }
    output.emplace_back(std::move(values));
  }
  return true;
}

bool append_record(Bridge &bridge, const RecordValues &values,
                   ldms_mval_t records_list) {
  auto &layout = bridge.layout;
  ldms_mval_t record = ldms_record_alloc(bridge.set, layout.record_type_idx);
  if (!record)
    return false;

  for (size_t i = 0; i < bridge.field_count; ++i) {
    switch (bridge.fields[i].type) {
    case LDMS_V_U64:
      ldms_record_set_u64(record, layout.field_idx[i], values[i].u64);
      break;
    case LDMS_V_S64:
      ldms_record_set_s64(record, layout.field_idx[i], values[i].s64);
      break;
    case LDMS_V_D64:
      ldms_record_set_double(record, layout.field_idx[i], values[i].d64);
      break;
    case LDMS_V_CHAR_ARRAY:
      ldms_record_array_set_str(record, layout.field_idx[i],
                                values[i].string.c_str());
      break;
    default:
      return false;
    }
  }
  return ldms_list_append_record(bridge.set, records_list, record) == 0;
}

int message_cb(ldms_msg_event_t event, void *user_data) {
  auto *bridge = static_cast<Bridge *>(user_data);
  if (!bridge || event->type != LDMS_MSG_EVENT_RECV)
    return 0;
  if (event->recv.type != LDMS_MSG_JSON || !event->recv.json)
    return EINVAL;

  json_entity_t message = event->recv.json;
  json_entity_t schema = required_value(message, "schema", JSON_STRING_VALUE);
  json_entity_t producer =
      required_value(message, "producer", JSON_STRING_VALUE);
  json_entity_t job_id = required_value(message, "job_id", JSON_STRING_VALUE);
  if (!schema || !producer || !job_id ||
      std::string{json_value_cstr(schema)} != bridge->schema_name)
    return EINVAL;

  uint64_t pid = 0;
  uint64_t window_start_ns = 0;
  uint64_t window_end_ns = 0;
  uint64_t window_sequence = 0;
  uint64_t chunk_index = 0;
  uint64_t chunk_count = 0;
  if (!read_u64(message, "pid", pid) ||
      !read_u64(message, "window_start_ns", window_start_ns) ||
      !read_u64(message, "window_end_ns", window_end_ns) ||
      !read_u64(message, "window_sequence", window_sequence) ||
      !read_u64(message, "chunk_index", chunk_index) ||
      !read_u64(message, "chunk_count", chunk_count))
    return EINVAL;

  std::vector<RecordValues> records;
  if (!parse_records(*bridge, message, records))
    return EINVAL;

  std::lock_guard<std::mutex> lock{bridge->update_mutex};
  auto &layout = bridge->layout;
  ldms_transaction_begin(bridge->set);
  ldms_mval_t records_list = ldms_metric_get(bridge->set, layout.records);
  int rc = records_list ? 0 : EIO;
  if (rc == 0)
    rc = ldms_list_purge(bridge->set, records_list);
  if (rc == 0) {
    ldms_metric_array_set_str(bridge->set, layout.producer,
                              json_value_cstr(producer));
    ldms_metric_array_set_str(bridge->set, layout.job_id,
                              json_value_cstr(job_id));
    ldms_metric_set_u64(bridge->set, layout.pid, pid);
    ldms_metric_set_u64(bridge->set, layout.window_start_ns, window_start_ns);
    ldms_metric_set_u64(bridge->set, layout.window_end_ns, window_end_ns);
    ldms_metric_set_u64(bridge->set, layout.window_sequence, window_sequence);
    ldms_metric_set_u64(bridge->set, layout.chunk_index, chunk_index);
    ldms_metric_set_u64(bridge->set, layout.chunk_count, chunk_count);

    for (const auto &values : records) {
      if (!append_record(*bridge, values, records_list)) {
        rc = ENOMEM;
        break;
      }
    }
    ldms_metric_modify(bridge->set, layout.records);
  }
  ldms_transaction_end(bridge->set);
  return rc;
}

void configure_string(attr_value_list *avl, const char *name,
                      std::string &output) {
  if (char *value = av_value(avl, name))
    output = value;
}

bool start_bridge(Bridge &bridge) {
  if (!create_schema_and_set(bridge))
    return false;
  bridge.msg_client =
      ldms_msg_subscribe(bridge.message_tag.c_str(), 0, message_cb, &bridge,
                         "ROCProfiler fixed-schema window receiver");
  if (!bridge.msg_client)
    return false;

  ovis_log(bridge.log, OVIS_LINFO,
           "rocprof_msg_sampler: schema=%s tag=%s instance=%s\n",
           bridge.schema_name, bridge.message_tag.c_str(),
           bridge.instance.c_str());
  return true;
}

int config(ldmsd_plug_handle_t, attr_value_list *, attr_value_list *avl) {
  std::string producer = "rocm_sampler";
  configure_string(avl, "producer", producer);
  g_pc.producer = g_device.producer = g_trace.producer = producer;

  /* instance/message_tag remain aliases for the original PC config. */
  configure_string(avl, "instance", g_pc.instance);
  configure_string(avl, "message_tag", g_pc.message_tag);
  configure_string(avl, "pc_instance", g_pc.instance);
  configure_string(avl, "device_instance", g_device.instance);
  configure_string(avl, "trace_instance", g_trace.instance);
  configure_string(avl, "pc_message_tag", g_pc.message_tag);
  configure_string(avl, "device_message_tag", g_device.message_tag);
  configure_string(avl, "trace_message_tag", g_trace.message_tag);

  if (char *value = av_value(avl, "max_records")) {
    const auto max_records =
        static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    if (max_records == 0)
      return EINVAL;
    g_pc.max_records = g_device.max_records = g_trace.max_records = max_records;
  }

  if (!ldms_msg_is_enabled()) {
    ovis_log(g_pc.log, OVIS_LERROR,
             "rocprof_msg_sampler: add msg_enable to sampler.conf\n");
    return ENOSYS;
  }

  if (!start_bridge(g_pc) || !start_bridge(g_device) || !start_bridge(g_trace))
    return errno ? errno : EIO;
  return 0;
}

int sample(ldmsd_plug_handle_t) { return 0; }

int constructor(ldmsd_plug_handle_t handle) {
  g_pc.log = g_device.log = g_trace.log = ldmsd_plug_log_get(handle);
  return 0;
}

void stop_bridge(Bridge &bridge) {
  if (bridge.msg_client) {
    ldms_msg_client_close(bridge.msg_client);
    bridge.msg_client = nullptr;
  }
  if (bridge.set) {
    ldms_set_unpublish(bridge.set);
    ldms_set_delete(bridge.set);
    bridge.set = nullptr;
  }
  if (bridge.layout.record_def) {
    ldms_record_delete(bridge.layout.record_def);
    bridge.layout.record_def = nullptr;
  }
  if (bridge.schema) {
    ldms_schema_delete(bridge.schema);
    bridge.schema = nullptr;
  }
}

void destructor(ldmsd_plug_handle_t) {
  stop_bridge(g_trace);
  stop_bridge(g_device);
  stop_bridge(g_pc);
}

} // namespace

extern "C" {
ldmsd_sampler ldmsd_plugin_interface = {
    .base =
        {
            .type = LDMSD_PLUGIN_SAMPLER,
            .config = config,
            .usage = usage,
            .constructor = constructor,
            .destructor = destructor,
        },
    .sample = sample,
};
}
