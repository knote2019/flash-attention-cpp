#include "gtest/gtest.h"
#include "tester.h"
#include "../ops/flash_attn_v2.h"


TEST(flash_attn_v2, test) {
    int b = 2;
    int sq = 256;
    int sk = 256;
    int hq = 32;
    int hk = 32;
    int d = 128;
    bool is_causal = true;
    int num_splits = 0;
    bool is_alibi = false;
    bool is_hybrid = false;
    int prefill_fraction = 0;
    int warmup_iterations = 1;
    int profiling_iterations = 10;
    int sleep_duration = 100;
    bool enable_check = false;

    cudaDeviceProp dev_prop{};
    FAI_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, 0));

    cudaStream_t stream{};

    Tester tester(b, sq, sk, hq, hk, d, is_causal, num_splits,
                  is_alibi, is_hybrid, prefill_fraction, stream, &dev_prop, warmup_iterations,
                  profiling_iterations, sleep_duration, enable_check);
    tester.evaluate(flash_attn_v2, "Flash-Attention-V2");

    std::cout << std::endl;
}