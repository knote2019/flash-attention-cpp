#include "ops/flash_attn_v2.h"
#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

DEFINE_uint32(b, 2, "batch size");
DEFINE_uint32(sq, 256, "q seq len");
DEFINE_uint32(sk, 256, "kv seq len");
DEFINE_uint32(hq, 32, "q head num");
DEFINE_uint32(hk, 32, "kv head num");
DEFINE_uint32(d, 128, "head dim");
DEFINE_bool(is_causal, true, "causal mask");
DEFINE_int32(num_splits, 0, "num splits of seq q len for flash attn");
DEFINE_bool(is_alibi, false, "enable alibi");
DEFINE_bool(is_hybrid, false, "hybrid mode");
DEFINE_uint32(prefill_fraction, 0, "percentage occupied by prefill in hybrid mode, the value ranges from 0 to 100");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the CPU result");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    cudaDeviceProp dev_prop;
    FAI_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));

    cudaStream_t stream = nullptr;

    FLOG(
            "FMHA: Softmax (Q (%u x %u x %u x %u) * K^T (%u x %u x %u x %u)) * V (%u x %u x %u x %u) = O (%u x %u x %u x "
            "%u)",
            FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk,
            FLAGS_d, FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d);
    FLOG(
            "Profiling: is causal: %d, num splits: %d, stream: %p, is alibi: %d, is hybrid: %d, prefill fraction: %u, "
            "warmup iterations: %u, profiling iterations: %u, sleep duration: %u ms, enable check: %d",
            FLAGS_is_causal, FLAGS_num_splits, stream, FLAGS_is_alibi, FLAGS_is_hybrid, FLAGS_prefill_fraction,
            FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);

    Tester tester(FLAGS_b, FLAGS_sq, FLAGS_sk, FLAGS_hq, FLAGS_hk, FLAGS_d, FLAGS_is_causal, FLAGS_num_splits,
                  FLAGS_is_alibi, FLAGS_is_hybrid, FLAGS_prefill_fraction, stream, &dev_prop, FLAGS_warmup_iterations,
                  FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);
    tester.evaluate(flash_attn_v2, "Flash-Attention-V2");

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    FLOG("Done");

    return 0;
}
