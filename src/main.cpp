
#include <torch/torch.h>

#include <fstream>
#include <memory>

#include "common.h"

// cuda context.
#include <ATen/cuda/CUDAContext.h>

int main(int argc, char** argv) {
    // init context.
    at::init();
    auto warp_size = at::cuda::warp_size();
    std::cout << "warp_size = " << warp_size << std::endl;

    // check cuda.
    std::cout << std::boolalpha;
    auto cuda_ready = torch::cuda::is_available();
    std::cout << "cuda_ready = " << cuda_ready << std::endl;
    if (!cuda_ready) {
        throw std::runtime_error("cuda not ready !!!");
    }

    int result;
    try {
        testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    } catch (std::exception& e) {
        std::cout << "Exception = " << e.what() << std::endl;
        return 1;
    }
    return result;
}
