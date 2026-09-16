#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

/*
Generic thread-safe window that can be utilized by each ROCProfiler collection service

A Window<Record> stores raw raw records as they arrive from their respective service callback. 
A timer periodically calls rotate() to close the current window and transfer its records into a WindowSnapshot<Record>
to be published into LDMS.

The class is templated so the same windowing logic can be reused for PcRecord, TraceRecord, and DeviceRecord without duplicating code.
*/

namespace rocm_ldms{
    template <typename Record>
    struct WindowSnapshot
    {
        uint64_t start_ns = 0;
        uint64_t end_ns = 0;
        uint64_t sequence = 0;
        std::vector<Record> records{};
    };

    template <typename Record>
    class Window{
    private:
        static uint64_t now_ns(){
            return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
            }
        //Leading underscore indicates the variable is a private member variable
        std::mutex mutex_{};
        std::vector<Record> records_{};
        uint64_t start_ns_ = 0;
        uint64_t sequence_ = 0;
    public:
        Window() : start_ns_{now_ns()} {}

        void reset(){
            std::lock_guard<std::mutex> lock{mutex_};
            records_.clear();
            start_ns_ = now_ns();
            sequence_ = 0;
        }

        void add(Record record){
            std::lock_guard<std::mutex> lock{mutex_};
            records_.push_back(std::move(record));
        }

        
        WindowSnapshot<Record> rotate() //lock_guard uses RAII which automatically unlocks when the object goes out of scope
        {
            //before:
            
            // ACTIVE WINDOW
            // [a][b][c][d]

            // rotate()

            // after:

            // SNAPSHOT                NEW ACTIVE WINDOW
            // [a][b][c][d]            []
            
            std::lock_guard<std::mutex> lock{mutex_};
            //The mutex here is a protective measure since swapping and adding records can create a potential race condition and corrupt the state

            const uint64_t cut_ns = now_ns();
            WindowSnapshot<Record> snapshot{};
            snapshot.start_ns = start_ns_;
            snapshot.end_ns = cut_ns;
            snapshot.sequence = sequence_++;
            snapshot.records.swap(records_);
            start_ns_ = cut_ns;
            return snapshot;
        }
    };
} //namespace rocm_ldms