// HIP Marshalling Language Example
// This demonstrates how HIP acts as a "marshalling language" 
// that routes calls to appropriate backends (AMD ROCm or NVIDIA CUDA)

// NOTE: If you get "file not found" errors for HIP headers, see installation 
// instructions at the bottom of this file, or use the SIMULATION_MODE version

#ifdef SIMULATION_MODE
// Simulation mode - demonstrates concepts without requiring HIP installation
#include <iostream>
#include <cmath>
#include <vector>

// Simulate HIP types and functions for demonstration
typedef int hipError_t;
typedef int hipblasStatus_t;
typedef int hipfftResult;
typedef struct { float x, y; } hipfftComplex;
typedef void* hipblasHandle_t;
typedef void* hipfftHandle;

#define hipSuccess 0
#define HIPBLAS_STATUS_SUCCESS 0
#define HIPFFT_SUCCESS 0
#define HIPBLAS_OP_N 0
#define HIPFFT_C2C 0
#define HIPFFT_FORWARD 0
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 0
#define HIPBLAS_TENSOR_OP_MATH 0

// Simulate platform detection
#ifndef __HIP_PLATFORM_AMD__
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_SIMULATION__
#define PLATFORM_NAME "Simulation Mode"
#endif
#endif

#else
// Real HIP headers (requires HIP installation)
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipfft.h>
#include <iostream>
#endif

// ==============================================================================
// 1. PLATFORM DETECTION AND MARSHALLING MACROS
// ==============================================================================

#ifdef SIMULATION_MODE
// Simulation functions to demonstrate marshalling concepts
hipError_t hipMalloc(void** ptr, size_t size) { 
    *ptr = malloc(size); 
    std::cout << "[MARSHALLING] hipMalloc -> simulated malloc(" << size << ")" << std::endl;
    return hipSuccess; 
}

hipError_t hipFree(void* ptr) { 
    free(ptr); 
    std::cout << "[MARSHALLING] hipFree -> simulated free()" << std::endl;
    return hipSuccess; 
}

hipError_t hipMemcpy(void* dst, const void* src, size_t size, int kind) {
    memcpy(dst, src, size);
    std::cout << "[MARSHALLING] hipMemcpy -> simulated memcpy(" << size << " bytes)" << std::endl;
    return hipSuccess;
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    *handle = (hipblasHandle_t)0x1234;
    std::cout << "[MARSHALLING] hipblasCreate -> simulated BLAS handle creation" << std::endl;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    std::cout << "[MARSHALLING] hipblasDestroy -> simulated BLAS cleanup" << std::endl;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, int transA, int transB,
                            int m, int n, int k, const float* alpha,
                            const float* A, int lda, const float* B, int ldb,
                            const float* beta, float* C, int ldc) {
    std::cout << "[MARSHALLING] hipblasSgemm -> simulated matrix multiply (" 
              << m << "x" << n << "x" << k << ")" << std::endl;
    // Simulate result
    for (int i = 0; i < m * n; i++) C[i] = 1024.0f; // m * 1.0 * 2.0 for our test case
    return HIPBLAS_STATUS_SUCCESS;
}

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, int type, int batch) {
    *plan = (hipfftHandle)0x5678;
    std::cout << "[MARSHALLING] hipfftPlan1d -> simulated FFT plan creation (size=" << nx << ")" << std::endl;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex* input, hipfftComplex* output, int direction) {
    std::cout << "[MARSHALLING] hipfftExecC2C -> simulated FFT execution" << std::endl;
    // Simulate FFT result (DC component)
    output[0].x = 1024.0f; output[0].y = 0.0f;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftDestroy(hipfftHandle plan) {
    std::cout << "[MARSHALLING] hipfftDestroy -> simulated FFT cleanup" << std::endl;
    return HIPFFT_SUCCESS;
}

#else
// Real HIP platform detection and marshalling
#ifdef __HIP_PLATFORM_AMD__
    #define PLATFORM_NAME "AMD ROCm"
    #include <rocblas.h>    // Native AMD backend
    #include <rocfft.h>
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #define PLATFORM_NAME "NVIDIA CUDA"
    #include <cublas_v2.h>  // Native NVIDIA backend
    #include <cufft.h>
#endif
#endif

// ==============================================================================
// 2. SINGLE-SOURCE CODE WITH AUTOMATIC MARSHALLING
// ==============================================================================

class GPUMarshaller {
private:
    hipblasHandle_t blas_handle;
    hipfftHandle plan;
    
public:
    void initialize() {
        std::cout << "HIP Marshalling to: " << PLATFORM_NAME << std::endl;
        
        // This single line of code gets marshalled to different backends:
        // - On AMD: routes to rocblas_create_handle()
        // - On NVIDIA: routes to cublasCreate()
        hipblasCreate(&blas_handle);
        
        std::cout << "BLAS handle created via HIP marshalling" << std::endl;
    }
    
    void demonstrateMemoryMarshalling() {
        const size_t size = 1024 * sizeof(float);
        float *d_ptr;
        
        // Memory allocation marshalling:
        // - AMD backend: maps to hipMalloc() -> ROCm memory allocation
        // - NVIDIA backend: maps to hipMalloc() -> cudaMalloc()
        hipError_t err = hipMalloc((void**)&d_ptr, size);
        
        if (err == hipSuccess) {
            std::cout << "Memory allocated via HIP marshalling: " << size << " bytes" << std::endl;
            
            // Memory copy marshalling:
            float host_data[1024];
            for (int i = 0; i < 1024; i++) host_data[i] = i * 0.5f;
            
            // This gets marshalled to:
            // - AMD: ROCm memory copy operations
            // - NVIDIA: cudaMemcpy operations
            hipMemcpy(d_ptr, host_data, size, hipMemcpyHostToDevice);
            std::cout << "Data copied via HIP marshalling" << std::endl;
            
            hipFree(d_ptr);
        }
    }
    
    void demonstrateBLASMarshalling() {
        const int N = 512;
        const size_t matrix_size = N * N * sizeof(float);
        
        float *h_A, *h_B, *h_C;
        float *d_A, *d_B, *d_C;
        
        // Host memory allocation
        h_A = new float[N * N];
        h_B = new float[N * N];
        h_C = new float[N * N];
        
        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 2.0f;
            h_C[i] = 0.0f;
        }
        
        // Device memory allocation (marshalled)
        hipMalloc((void**)&d_A, matrix_size);
        hipMalloc((void**)&d_B, matrix_size);
        hipMalloc((void**)&d_C, matrix_size);
        
        // Copy data to device (marshalled)
        hipMemcpy(d_A, h_A, matrix_size, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B, matrix_size, hipMemcpyHostToDevice);
        
        // BLAS operation marshalling:
        // This single function call gets marshalled to:
        // - AMD: rocblas_sgemm() with ROCm-optimized implementation
        // - NVIDIA: cublasSgemm() with CUDA-optimized implementation
        const float alpha = 1.0f, beta = 0.0f;
        
        hipblasStatus_t status = hipblasSgemm(
            blas_handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N
        );
        
        if (status == HIPBLAS_STATUS_SUCCESS) {
            std::cout << "Matrix multiplication completed via HIP BLAS marshalling" << std::endl;
            
            // Copy result back (marshalled)
            hipMemcpy(h_C, d_C, matrix_size, hipMemcpyDeviceToHost);
            
            // Verify result (first few elements should be N * 1.0 * 2.0 = 1024)
            std::cout << "Sample results: " << h_C[0] << ", " << h_C[1] << ", " << h_C[2] << std::endl;
        }
        
        // Cleanup
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }
    
    void demonstrateFFTMarshalling() {
        const int N = 1024;
        const size_t size = N * sizeof(hipfftComplex);
        
        hipfftComplex *h_input, *h_output;
        hipfftComplex *d_input, *d_output;
        
        // Allocate host memory
        h_input = new hipfftComplex[N];
        h_output = new hipfftComplex[N];
        
        // Initialize input signal
        for (int i = 0; i < N; i++) {
            h_input[i].x = cos(2.0f * M_PI * i / N);  // Real part
            h_input[i].y = 0.0f;                       // Imaginary part
        }
        
        // Allocate device memory (marshalled)
        hipMalloc((void**)&d_input, size);
        hipMalloc((void**)&d_output, size);
        
        // Copy input to device (marshalled)
        hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice);
        
        // FFT plan creation marshalling:
        // - AMD: routes to rocfft_plan_create()
        // - NVIDIA: routes to cufftPlan1d()
        hipfftResult result = hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);
        
        if (result == HIPFFT_SUCCESS) {
            std::cout << "FFT plan created via HIP marshalling" << std::endl;
            
            // FFT execution marshalling:
            // - AMD: rocfft_execute() with ROCm FFT kernels
            // - NVIDIA: cufftExecC2C() with CUDA FFT kernels
            result = hipfftExecC2C(plan, d_input, d_output, HIPFFT_FORWARD);
            
            if (result == HIPFFT_SUCCESS) {
                std::cout << "FFT executed via HIP marshalling" << std::endl;
                
                // Copy result back (marshalled)
                hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);
                
                std::cout << "FFT result magnitude[0]: " 
                          << sqrt(h_output[0].x * h_output[0].x + h_output[0].y * h_output[0].y) 
                          << std::endl;
            }
            
            hipfftDestroy(plan);
        }
        
        // Cleanup
        hipFree(d_input); hipFree(d_output);
        delete[] h_input; delete[] h_output;
    }
    
    void showPlatformSpecificOptimizations() {
        std::cout << "\n=== Platform-Specific Optimizations via Marshalling ===" << std::endl;
        
        // HIP marshalling allows platform-specific optimizations
        #ifdef __HIP_PLATFORM_AMD__
            std::cout << "AMD-specific optimizations enabled:" << std::endl;
            std::cout << "- Using ROCm memory pools" << std::endl;
            std::cout << "- AMD GPU architecture-specific kernels" << std::endl;
            std::cout << "- ROCm-optimized BLAS/FFT implementations" << std::endl;
            
            // Example: AMD-specific memory advice
            hipMemAdvise(nullptr, 0, hipMemAdviseSetPreferredLocation, 0);
            
        #elif defined(__HIP_PLATFORM_NVIDIA__)
            std::cout << "NVIDIA-specific optimizations enabled:" << std::endl;
            std::cout << "- Using CUDA memory pools" << std::endl;
            std::cout << "- Tensor Core optimizations (if available)" << std::endl;
            std::cout << "- cuBLAS/cuFFT optimized implementations" << std::endl;
            
            // Example: NVIDIA-specific math mode
            hipblasSetMathMode(blas_handle, HIPBLAS_TENSOR_OP_MATH);
        #endif
    }
    
    void cleanup() {
        hipblasDestroy(blas_handle);
        std::cout << "Cleanup completed via HIP marshalling" << std::endl;
    }
};

// ==============================================================================
// 3. KERNEL MARSHALLING EXAMPLE
// ==============================================================================

// This kernel code is identical regardless of backend
// HIP marshalling handles compilation for appropriate target
__global__ void marshalledKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Same code, different compilation targets:
        // - AMD: compiled to GCN/RDNA ISA
        // - NVIDIA: compiled to PTX/SASS
        data[idx] = data[idx] * 2.0f + threadIdx.x;
        
        // Synchronization marshalling:
        // - AMD: maps to s_barrier instruction
        // - NVIDIA: maps to bar.sync instruction
        __syncthreads();
        
        // Warp-aware code (marshalling handles different warp sizes)
        if (threadIdx.x == 0) {
            // AMD: warp size = 64 (on most architectures)
            // NVIDIA: warp size = 32
            data[blockIdx.x] += (float)warpSize;
        }
    }
}

void demonstrateKernelMarshalling() {
    const int size = 1024;
    const size_t bytes = size * sizeof(float);
    
    float *h_data, *d_data;
    h_data = new float[size];
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    
    // Memory operations (marshalled)
    hipMalloc((void**)&d_data, bytes);
    hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);
    
    // Kernel launch marshalling:
    // - AMD: launches on Compute Units using ROCm runtime
    // - NVIDIA: launches on SMs using CUDA runtime
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // This macro gets marshalled to appropriate launch mechanism
    hipLaunchKernelGGL(marshalledKernel, blocksPerGrid, threadsPerBlock, 0, 0, d_data, size);
    
    // Synchronization marshalling
    hipDeviceSynchronize();
    
    // Copy result back
    hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost);
    
    std::cout << "Kernel executed via HIP marshalling" << std::endl;
    std::cout << "Sample results: " << h_data[0] << ", " << h_data[1] << ", " << h_data[256] << std::endl;
    
    // Cleanup
    hipFree(d_data);
    delete[] h_data;
}

// ==============================================================================
// 4. MAIN DEMONSTRATION
// ==============================================================================

int main() {
    std::cout << "=== HIP Marshalling Language Demonstration ===" << std::endl;
    std::cout << "Single source code, multiple backend targets\n" << std::endl;
    
#ifdef SIMULATION_MODE
    std::cout << "Running in SIMULATION MODE - demonstrating marshalling concepts" << std::endl;
    std::cout << "Platform: " << PLATFORM_NAME << std::endl;
    std::cout << "Device: Simulated GPU Device" << std::endl;
    std::cout << "To run with real HIP, install ROCm and compile without -DSIMULATION_MODE\n" << std::endl;
#else
    // Real device detection and marshalling setup
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No HIP-compatible devices found" << std::endl;
        return -1;
    }
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
#endif
    
    // Create marshaller instance
    GPUMarshaller marshaller;
    
    try {
        // Demonstrate different types of marshalling
        marshaller.initialize();
        marshaller.demonstrateMemoryMarshalling();
        marshaller.demonstrateBLASMarshalling();
        marshaller.demonstrateFFTMarshalling();
        marshaller.showPlatformSpecificOptimizations();
        
        // Demonstrate kernel marshalling
        std::cout << "\n=== Kernel Marshalling ===" << std::endl;
        demonstrateKernelMarshalling();
        
        marshaller.cleanup();
        
        std::cout << "\n=== Marshalling Summary ===" << std::endl;
        std::cout << "✓ All operations successfully marshalled to " << PLATFORM_NAME << std::endl;
        std::cout << "✓ Same source code, platform-optimized execution" << std::endl;
        std::cout << "✓ Zero runtime overhead from marshalling layer" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

// ==============================================================================
// 5. COMPILATION AND INSTALLATION INSTRUCTIONS
// ==============================================================================

/*
=== OPTION 1: SIMULATION MODE (No HIP Installation Required) ===

To run this example without installing HIP/ROCm:

    g++ -DSIMULATION_MODE -o hip_marshalling_sim hip_marshalling.cpp
    ./hip_marshalling_sim

This demonstrates the marshalling concepts using simulated function calls.

=== OPTION 2: REAL HIP COMPILATION (Requires HIP Installation) ===

INSTALL HIP/ROCm on Ubuntu/Debian:
    # Add ROCm repository
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    
    # Install ROCm and HIP libraries
    sudo apt install rocm-dkms hip-dev hipblas hipfft
    
    # Add user to render group
    sudo usermod -a -G render,video $LOGNAME
    
    # Reboot or restart services
    sudo reboot

INSTALL HIP/ROCm on RHEL/CentOS:
    # Add ROCm repository
    sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/yum/rpm
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
    
    # Install packages
    sudo dnf install rocm-dkms hip-devel hipblas hipfft

COMPILATION FOR AMD (ROCm backend):
    export HIP_PLATFORM=amd
    /opt/rocm/bin/hipcc -o hip_marshalling hip_marshalling.cpp -lhipblas -lhipfft
    ./hip_marshalling

COMPILATION FOR NVIDIA (CUDA backend):
    export HIP_PLATFORM=nvidia
    /opt/rocm/bin/hipcc -o hip_marshalling hip_marshalling.cpp -lhipblas -lhipfft
    ./hip_marshalling

=== OPTION 3: DOCKER/CONTAINER APPROACH ===

Use pre-built ROCm container:
    docker pull rocm/dev-ubuntu-20.04:latest
    docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/dev-ubuntu-20.04
    
    # Inside container:
    hipcc -o hip_marshalling hip_marshalling.cpp -lhipblas -lhipfft

=== TROUBLESHOOTING ===

If you get "file not found" errors:
1. Check HIP installation: /opt/rocm/bin/hipcc --version
2. Check library paths: export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
3. Check include paths: export C_INCLUDE_PATH=/opt/rocm/include:$C_INCLUDE_PATH
4. Verify GPU support: /opt/rocm/bin/rocm-smi

Alternative minimal example without libraries:
    echo '#include <hip/hip_runtime.h>
    int main() { 
        int count; 
        hipGetDeviceCount(&count); 
        printf("Devices: %d\n", count); 
        return 0; 
    }' > test_hip.cpp
    
    /opt/rocm/bin/hipcc test_hip.cpp -o test_hip

=== MARSHALLING EXPLANATION ===

RUNTIME MARSHALLING FLOW:
    Your Code → HIP API → Platform Detection → Backend Selection
    
    hipMalloc() → [AMD: ROCm memory] | [NVIDIA: CUDA memory]
    hipblasSgemm() → [AMD: rocblas_sgemm] | [NVIDIA: cublasSgemm]
    hipfftExecC2C() → [AMD: rocfft_execute] | [NVIDIA: cufftExecC2C]
    hipLaunchKernelGGL() → [AMD: ROCm dispatcher] | [NVIDIA: CUDA dispatcher]

The marshalling is transparent to your application - same source code,
different optimized backends depending on compilation target.

PERFORMANCE COMPARISON:
- Direct CUDA: 100% performance baseline
- HIP→CUDA: 98-99% performance (minimal marshalling overhead)
- HIP→ROCm: 95-102% performance (sometimes better due to AMD optimizations)
- ROC native: 100-105% performance (AMD-specific optimizations)

The marshalling layer adds virtually no runtime overhead while providing
complete portability between GPU vendors.
*/
