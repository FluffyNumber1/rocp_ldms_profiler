# Fixed ROCProfiler-to-LDMS schemas

`rocprof_msg_sampler.cpp` intentionally has no runtime schema inference. It
contains three visible `FieldSpec` tables (`PC_FIELDS`, `DEVICE_FIELDS`, and
`TRACE_FIELDS`), creates one set per table, and subscribes each set to its own
message tag.

Every callback performs the same small operation: verify its known JSON
schema, parse only the fixed fields, and purge/refill that set's `records` list
inside one LDMS transaction.

The aggregator decomposes those lists with:

- `pc-decomp.json` -> `pc_sampling/rocm_pc_sample`
- `device-decomp.json` -> `device_counting/rocm_device_counter`
- `trace-decomp.json` -> `tracing/rocm_trace_event`

Device data uses long format so different configured counter names do not
require different LDMS schemas. Trace data uses a normalized superset; the
`valid_fields` bit mask says which optional values apply to each event kind.

Run `ldmsd` from the repository root so the decomposition paths in
`aggregator.conf` resolve. The three `test-*-message.json` files can be sent
with `ldms_msg_publish` to test the full sampler/aggregator/CSV half without a
GPU workload; the exact commands are in the top-level README.
