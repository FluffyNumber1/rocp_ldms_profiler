# ROCProfiler SDK data through LDMS

A configurable ROCProfiler SDK tool that sends raw PC samples, device counter
readings, and application trace records to LDMS.

The tool uses two libraries:

1. `build/librocprof_ldms_tool.so` runs inside a HIP application and collects
   the enabled ROCProfiler SDK modes.
2. `ldms/librocprof_msg_sampler.so` runs inside the sampler `ldmsd` and turns
   each message tag into a fixed LDMS metric set.

The libraries meet through the LDMS message connection on port `10411`. The
aggregator reads three independent metric sets and writes decomposed CSV rows.

| Mode | Tool source | Message tag | LDMS schema | CSV container/file |
|---|---|---|---|---|
| PC sampling | `src/pc.cpp` | `rocm_pc_raw` | `rocm_pc_raw_window` | `pc_sampling/rocm_pc_sample` |
| Device counters | `src/device.cpp` | `rocm_device_raw` | `rocm_device_raw_window` | `device_counting/rocm_device_counter` |
| Application trace | `src/trace.cpp` | `rocm_trace_raw` | `rocm_trace_raw_window` | `tracing/rocm_trace_event` |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Tool lifecycle, configuration, publication, and PC/device/trace collectors |
| `include/` | Record types, shared window buffering, and collector interfaces |
| `config/` | Runtime INI configuration |
| `ldms/` | Receiving plugin, LDMS configuration, CSV decomposition, and message fixtures |
| `vector_add.cpp` | Small HIP example for verifying collection |

## Build integration

This repository contains the tool source and runtime configuration. Build recipes
and installation paths are maintained separately for each target system.
Compile the tool against ROCProfiler SDK and LDMS, and the receiving plugin
against LDMS and its OVIS dependencies.

The runtime examples below assume the libraries use the paths listed above and
the HIP example has been compiled to `build/vector_add`.

For an uninstalled LDMS plugin:

```sh
export LDMSD_PLUGIN_LIBPATH="$PWD/ldms${LDMSD_PLUGIN_LIBPATH:+:$LDMSD_PLUGIN_LIBPATH}"
```

Start the sampler and aggregator with the existing config files in the normal
way. Run them from the repository root so `ldms/*-decomp.json` resolves.

## Select a collection mode

Edit `config/rocprofiler.conf`. PC and device counters are deliberately
separate test modes because they compete for GPU performance-monitoring
resources. Trace can be enabled with either mode.

PC test:

```ini
[pc]
enabled=true

[device]
enabled=false

[trace]
enabled=false
```

Device-counter test:

```ini
[pc]
enabled=false

[device]
enabled=true
interval_ms=50
counters=SQ_WAVES

[trace]
enabled=false
```

Trace test:

```ini
[pc]
enabled=false

[device]
enabled=false

[trace]
enabled=true
```

The device CSV is long format: one row per returned counter instance. That
means the `counters=` list can change without rebuilding the plugin or changing
the LDMS schema. `value_valid=0` means ROCProfiler returned a non-finite value;
the CSV value is then written as `0` so the JSON message stays valid.

## Run the HIP workload

```sh
export ROCM_LDMS_CONFIG="$PWD/config/rocprofiler.conf"
export ROCP_TOOL_LIBRARIES="$PWD/build/librocprof_ldms_tool.so"
export ROCPROFILER_LOG_LEVEL=info

./build/vector_add
```

For PC mode only, also set:

```sh
export ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1
```

## Test the LDMS/CSV half without a GPU

With the sampler and aggregator running, publish any of the fixed test
messages:

```sh
ldms_msg_publish -x sock -h 127.0.0.1 -p 10411 -a none \
  -m rocm_pc_raw -t json -f ldms/test-pc-message.json

ldms_msg_publish -x sock -h 127.0.0.1 -p 10411 -a none \
  -m rocm_device_raw -t json -f ldms/test-device-message.json

ldms_msg_publish -x sock -h 127.0.0.1 -p 10411 -a none \
  -m rocm_trace_raw -t json -f ldms/test-trace-message.json
```

CSV output appears below `/tmp/rocprof-ldms/pc_sampling`,
`/tmp/rocprof-ldms/device_counting`, and `/tmp/rocprof-ldms/tracing`.
