# ROCProfiler SDK data through LDMS

A configurable tool that collects PC samples, device counter readings, and
application trace records through the ROCProfiler SDK and sends them to LDMS.
Each collector retains individual decoded records, leaving aggregation and
analysis outside the collection code.

## Tool structure

The tool has two components. `librocprof_ldms_tool.so` collects profiling data
inside a HIP application. `librocprof_msg_sampler.so` receives that data in a
separate LDMS process and maps it into LDMS metric sets.

The collectors share configuration, window buffering, and message delivery,
while each collector handles its own SDK records.

| File or directory | Responsibility |
|---|---|
| `src/tool.cpp` | Coordinates startup, shutdown, enabled collectors, and the publication timer |
| `src/config.cpp` | Reads and validates the INI configuration |
| `src/pc.cpp` | Collects and decodes PC samples |
| `src/device.cpp` | Reads selected device counters at a configured interval |
| `src/trace.cpp` | Collects application trace events |
| `src/publisher.cpp` | Manages the LDMS connection and sends JSON messages |
| `include/window.hpp` | Provides shared buffering and batch rotation for records |
| `include/` | Defines record types and component interfaces |
| `config/rocprofiler.conf` | Describes collection and delivery settings |
| `ldms/` | Contains the receiving plugin, metric-set configuration, and CSV field mappings |

## Collection modes

- **PC sampling** records sampled GPU program-counter information and associated
  execution details.
- **Device counting** records readings from selected GPU performance counters.
- **Application tracing** records events such as HIP runtime calls, kernel
  dispatches, memory copies, and memory allocations.

The current tool treats PC sampling and device counting as mutually exclusive
modes. Tracing can operate on its own or alongside either mode.

## Configuration

One INI file groups settings under `[tool]`, `[ldms]`, `[pc]`, `[device]`, and
`[trace]`. These sections describe the enabled collectors, counter selection,
sampling intervals, publication interval, message size, and LDMS connection.

Collection timing and publication timing are separate. Device counters have
a read interval in milliseconds, PC sampling has an interval in cycles, and
the shared publication timer controls when buffered records form a batch.

## Record windows and data flow

Each collector decodes SDK output into its record type and adds records to a
`Window<Record>`. The same window implementation supports `PcRecord`,
`DeviceRecord`, and `TraceRecord`.

When the publication timer fires, `rotate()` swaps the active record vector
into a `WindowSnapshot<Record>`. The active vector is then empty and ready for
new records. Adding and rotating use the same lock. Formatting the snapshot
as JSON happens after that lock is released.

The snapshot contains window start and end timestamps and a sequence number.
Individual record timestamps describe the observations, while window timestamps
describe the batch. SDK buffering can delay when records reach a window.

Collectors divide a batch into messages according to the configured record
limit. The publisher sends those JSON messages to the LDMS receiving plugin.
The plugin places records into separate metric sets for each collection mode,
and the aggregator maps their fields into CSV rows.

## Data formats

| Mode | Message tag | LDMS schema | CSV container/file |
|---|---|---|---|
| PC sampling | `rocm_pc_raw` | `rocm_pc_raw_window` | `pc_sampling/rocm_pc_sample` |
| Device counters | `rocm_device_raw` | `rocm_device_raw_window` | `device_counting/rocm_device_counter` |
| Application trace | `rocm_trace_raw` | `rocm_trace_raw_window` | `tracing/rocm_trace_event` |

Device counter output uses one row per returned counter instance, with the
counter name and value stored as fields. Different counter selections therefore
use the same LDMS schema. A non-finite counter value is represented as `0` with
`value_valid=0`, distinguishing it from a valid zero reading.
