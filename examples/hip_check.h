// SPDX-License-Identifier: MIT
// HIP Error Checking Utilities
#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Macro to check HIP API calls and report errors
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t error = call;                                                \
        if (error != hipSuccess) {                                              \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " - " << hipGetErrorString(error) << " (code "         \
                      << error << ")" << std::endl;                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Non-fatal version that just warns but doesn't exit
#define HIP_WARN(call)                                                          \
    do {                                                                        \
        hipError_t error = call;                                                \
        if (error != hipSuccess) {                                              \
            std::cerr << "HIP warning at " << __FILE__ << ":" << __LINE__       \
                      << " - " << hipGetErrorString(error) << " (code "         \
                      << error << ")" << std::endl;                             \
        }                                                                       \
    } while (0)

// Check last error (useful after kernel launches)
#define HIP_CHECK_LAST()                                                        \
    do {                                                                        \
        hipError_t error = hipGetLastError();                                   \
        if (error != hipSuccess) {                                              \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " - " << hipGetErrorString(error) << " (code "         \
                      << error << ")" << std::endl;                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Synchronize device and check for errors
#define HIP_CHECK_SYNC()                                                        \
    do {                                                                        \
        hipError_t error = hipDeviceSynchronize();                              \
        if (error != hipSuccess) {                                              \
            std::cerr << "HIP synchronization error at " << __FILE__ << ":"     \
                      << __LINE__ << " - " << hipGetErrorString(error)          \
                      << " (code " << error << ")" << std::endl;                \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
