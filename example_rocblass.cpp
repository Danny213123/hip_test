#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Test basic rocBLAS functionality
    rocblas_handle handle;
    rocblas_status status = rocblas_create_handle(&handle);
    
    if (status != rocblas_status_success) {
        std::cout << "Failed to create rocBLAS handle: " << status << std::endl;
        return 1;
    }
    
    std::cout << "rocBLAS handle created successfully!" << std::endl;
    
    // Test basic HIP functionality
    int deviceCount;
    hipError_t hipStatus = hipGetDeviceCount(&deviceCount);
    
    if (hipStatus != hipSuccess) {
        std::cout << "HIP error: " << hipGetErrorString(hipStatus) << std::endl;
        rocblas_destroy_handle(handle);
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " GPU device(s)" << std::endl;
    
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "GPU 0: " << prop.name << std::endl;
        std::cout << "Architecture: " << prop.gcnArchName << std::endl;
    }
    
    rocblas_destroy_handle(handle);
    std::cout << "Basic test completed successfully!" << std::endl;
    return 0;
}
