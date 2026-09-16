#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

namespace rocm_ldms {
// Opt-in aggregate diagnostics: no per-sample log writes. Durations overlap
// across threads and nested scopes; they must not be added as a time budget.
class Diagnostics {
public:
  struct Metric { unsigned long long calls = 0, total_ns = 0, max_ns = 0; };
  // SDK finalizers can execute after normal static destruction has begun.
  static Diagnostics &get() { static auto *instance = new Diagnostics; return *instance; }
  bool enabled() const { return !path_.empty(); }
  void add(const char *name, unsigned long long value) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock{mutex_};
    auto &m = metrics_[name];
    ++m.calls; m.total_ns += value;
    if (value > m.max_ns) m.max_ns = value;
  }
  void window(unsigned long long seq, size_t records, size_t chunks) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock{mutex_};
    std::ofstream out{path_ + ".windows.jsonl", std::ios::app};
    out << "{\"sequence\":" << seq << ",\"records\":" << records
        << ",\"chunks\":" << chunks << "}\n";
  }
  void save() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock{mutex_};
    std::ofstream out{path_};
    out << "{\"schema_version\":1,\"metrics\":{";
    bool first = true;
    for (const auto &entry : metrics_) {
      if (!first) out << ',';
      first = false;
      const auto &m = entry.second;
      out << '"' << entry.first << "\":{\"calls\":" << m.calls
          << ",\"total\":" << m.total_ns << ",\"max\":" << m.max_ns << '}';
    }
    out << "}}\n";
  }
private:
  Diagnostics() {
    const char *p = std::getenv("ROCM_LDMS_DIAGNOSTICS");
    if (p) path_ = p;
  }
  std::string path_;
  std::mutex mutex_;
  std::map<std::string, Metric> metrics_;
};

class DiagnosticTimer {
public:
  explicit DiagnosticTimer(const char *name) : name_{name}, enabled_{Diagnostics::get().enabled()} {
    if (enabled_) start_ = std::chrono::steady_clock::now();
  }
  ~DiagnosticTimer() {
    if (enabled_) Diagnostics::get().add(name_, static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count()));
  }
private:
  const char *name_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_{};
};
} // namespace rocm_ldms
