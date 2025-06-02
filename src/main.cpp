#include "gtest/gtest.h"

int main(int argc, char **argv) {
    int result;
    try {
        testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    } catch (std::exception &e) {
        std::cout << "Exception = " << e.what() << std::endl;
        return 1;
    }
    return result;
}
