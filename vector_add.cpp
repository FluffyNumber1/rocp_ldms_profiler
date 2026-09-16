#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t error = (call);                                       \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(error)            \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';       \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (false)

__global__ void vector_add(const float* a, const float* b, float* c,
                           std::size_t count)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    // Repeating a large vector addition keeps the GPU active long enough for
    // PC sampling. Pass a different iteration count as the first argument.
    int iterations = 20000;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0) {
            std::cerr << "Usage: " << argv[0] << " [positive-iterations]\n";
            return EXIT_FAILURE;
        }
    }

    constexpr std::size_t count = 1ULL << 24;
    const std::size_t bytes = count * sizeof(float);

    std::vector<float> host_a(count, 1.0F);
    std::vector<float> host_b(count, 2.0F);
    std::vector<float> host_c(count, 0.0F);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_a), bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_b), bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_c), bytes));
    HIP_CHECK(hipMemcpy(device_a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), bytes, hipMemcpyHostToDevice));

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (count + threads_per_block - 1) / threads_per_block);

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        vector_add<<<blocks, threads_per_block>>>(
            device_a, device_b, device_c, count);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipMemcpy(host_c.data(), device_c, bytes, hipMemcpyDeviceToHost));

    bool correct = true;
    for (const float value : host_c) {
        if (std::fabs(value - 3.0F) > 0.0001F) {
            correct = false;
            break;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_c));

    if (!correct) {
        std::cerr << "Vector-add verification FAILED\n";
        return EXIT_FAILURE;
    }

    std::cout << "Vector-add verification PASSED\n"
              << "Elements: " << count << '\n'
              << "Kernel launches: " << iterations << '\n'
              << "GPU time: " << elapsed_ms << " ms\n";
    return EXIT_SUCCESS;
}
