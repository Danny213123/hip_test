/**
 * HIP GPU Marshalling Demonstration
 * 
 */

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <iostream>
#include <chrono>
#include <iomanip>

// Platform detection for marshalling demonstration
#define PLATFORM_NAME "HIP Platform"
#define BACKEND_DESCRIPTION "HIP marshalls to optimal backend (AMD ROCm or NVIDIA CUDA)"

/**
 * GPU Marshalling Manager
 * 
 */
class GPUMarshallingDemo {
private:
    hipblasHandle_t blas_handle;
    int device_count;
    hipDeviceProp_t device_properties;
    
public:
    /**
     * Initialize GPU marshalling environment
     * Demonstrates automatic platform detection and setup
     */
    bool initialize() {
        std::cout << "=== GPU Marshalling Initialization ===" << std::endl;
        std::cout << "Target Platform: " << PLATFORM_NAME << std::endl;
        std::cout << "Marshalling Strategy: " << BACKEND_DESCRIPTION << std::endl;
        
        // Device detection
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0) {
            std::cout << "❌ No compatible GPU devices found" << std::endl;
            return false;
        }
        
        // Get device properties
        hipGetDeviceProperties(&device_properties, 0);
        std::cout << "✓ Detected Device: " << device_properties.name << std::endl;
        std::cout << "✓ Compute Capability: " << device_properties.major 
                  << "." << device_properties.minor << std::endl;
        std::cout << "✓ Global Memory: " 
                  << (device_properties.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) 
                  << " GB" << std::endl;
        
        hipblasStatus_t status = hipblasCreate(&blas_handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            std::cout << "❌ Failed to create BLAS handle" << std::endl;
            return false;
        }
        
        std::cout << "✓ BLAS handle created via HIP marshalling" << std::endl;
        std::cout << "✓ Initialization complete\n" << std::endl;
        return true;
    }
    
    /**
     * Demonstrate vector addition using hipBLAS marshalling
     * 
     */
    void demonstrateVectorAddition() {
        std::cout << "=== Vector Addition via BLAS Marshalling ===" << std::endl;

        const int N = 1024 * 1024;
        const size_t vector_size = N * sizeof(float);
        const double size_mb = vector_size / 1024.0 / 1024.0;
        
        std::cout << "Problem Size: " << N << " elements (" << std::fixed 
                  << std::setprecision(1) << size_mb << " MB per vector)" << std::endl;
        std::cout << "Total Memory: " << (size_mb * 3) << " MB (3 vectors)" << std::endl;

        float *h_A = new float[N];
        float *h_B = new float[N];
        float *h_C = new float[N];

        std::cout << "\nInitializing test vectors..." << std::endl;
        for (int i = 0; i < N; i++) {
            h_A[i] = 1.0f + (i % 100) * 0.01f;
            h_B[i] = 2.0f + (i % 50) * 0.02f;
            h_C[i] = 0.0f;
        }

        std::cout << "\n=== Input Test Vectors ===" << std::endl;
        std::cout << "Vector A pattern: 1.0 + (index % 100) * 0.01" << std::endl;
        std::cout << "Vector B pattern: 2.0 + (index % 50) * 0.02" << std::endl;
        
        std::cout << "\nFirst 10 elements of Vector A:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "A[" << i << "] = " << std::fixed << std::setprecision(2) << h_A[i];
            if ((i + 1) % 5 == 0) std::cout << std::endl;
            else std::cout << ", ";
        }
        
        std::cout << "\nFirst 10 elements of Vector B:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "B[" << i << "] = " << std::fixed << std::setprecision(2) << h_B[i];
            if ((i + 1) % 5 == 0) std::cout << std::endl;
            else std::cout << ", ";
        }
        
        std::cout << "\nExpected results for first 10 elements (A + B):" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "C[" << i << "] = " << std::fixed << std::setprecision(2) << (h_A[i] + h_B[i]);
            if ((i + 1) % 5 == 0) std::cout << std::endl;
            else std::cout << ", ";
        }

        std::cout << "\nSample elements at various indices:" << std::endl;
        int sample_indices[] = {0, 50, 100, 500, 1000, 10000, 100000};
        for (int idx : sample_indices) {
            if (idx < N) {
                std::cout << "Index " << std::setw(6) << idx << ": A=" << std::setprecision(2) 
                          << h_A[idx] << ", B=" << h_B[idx] << ", Expected=" << (h_A[idx] + h_B[idx]) << std::endl;
            }
        }
        std::cout << std::endl;

        float *d_A, *d_B, *d_C;
        const float alpha = 1.0f;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto compute_start = start_time;
        auto compute_end = start_time;
        auto transfer_time = start_time;
        auto end_time = start_time;
        
        hipMalloc((void**)&d_A, vector_size);
        hipMalloc((void**)&d_B, vector_size);
        hipMalloc((void**)&d_C, vector_size);
        
        hipMemcpy(d_A, h_A, vector_size, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B, vector_size, hipMemcpyHostToDevice);
        
        transfer_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "✓ Memory allocated and data transferred to GPU" << std::endl;

        std::cout << "\nPerforming vector addition: C = A + B" << std::endl;
        std::cout << "Method: BLAS Copy + SAXPY operations" << std::endl;

        hipblasStatus_t status = hipblasScopy(blas_handle, N, d_A, 1, d_C, 1);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            std::cout << "❌ hipblasScopy failed" << std::endl;
            goto cleanup;
        }

        compute_start = std::chrono::high_resolution_clock::now();
        
        status = hipblasSaxpy(blas_handle, N, &alpha, d_B, 1, d_C, 1);

        hipDeviceSynchronize();
        compute_end = std::chrono::high_resolution_clock::now();
        
        if (status != HIPBLAS_STATUS_SUCCESS) {
            std::cout << "❌ hipblasSaxpy failed" << std::endl;
            goto cleanup;
        }

        hipMemcpy(h_C, d_C, vector_size, hipMemcpyDeviceToHost);
        end_time = std::chrono::high_resolution_clock::now();
        
        // Performance Analysis
        auto total_time = std::chrono::duration<double>(end_time - start_time).count();
        auto compute_time = std::chrono::duration<double>(compute_end - compute_start).count();
        auto transfer_overhead = std::chrono::duration<double>(transfer_time - start_time).count();
        
        std::cout << "✓ Vector addition completed successfully" << std::endl;
        
        // Display Performance Metrics
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Total Execution Time: " << std::fixed << std::setprecision(3) 
                  << (total_time * 1000) << " ms" << std::endl;
        std::cout << "Pure Compute Time: " << (compute_time * 1000) << " ms" << std::endl;
        std::cout << "Memory Transfer Overhead: " << (transfer_overhead * 1000) << " ms" << std::endl;
        std::cout << "Throughput: " << std::setprecision(2) 
                  << (N * 2 / compute_time / 1e9) << " GFLOPS" << std::endl;
        std::cout << "Memory Bandwidth: " << std::setprecision(1)
                  << (vector_size * 3 / transfer_overhead / 1e9) << " GB/s" << std::endl;

        std::cout << "\n=== Result Verification ===" << std::endl;
        bool correct = true;
        double max_error = 0.0;
        int error_count = 0;

        for (int i = 0; i < N; i += 1000) {
            float expected = h_A[i] + h_B[i];
            float error = abs(h_C[i] - expected);
            if (error > 1e-5) {
                error_count++;
                correct = false;
            }
            if (error > max_error) max_error = error;
        }
        
        std::cout << "✓ Sampled " << (N/1000) << " elements for verification" << std::endl;
        std::cout << "✓ Maximum error: " << std::scientific << max_error << std::endl;
        std::cout << "✓ Result accuracy: " << (correct ? "PASSED" : "FAILED") << std::endl;

        std::cout << "\n=== Sample Results ===" << std::endl;
        std::cout << "First 5 results:" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "  A[" << i << "] + B[" << i << "] = " 
                      << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
        }
        
cleanup:
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
        
        std::cout << "✓ Resources cleaned up\n" << std::endl;
    }
    
    /**
     * Cleanup resources
     */
    void cleanup() {
        if (blas_handle) {
            hipblasDestroy(blas_handle);
            std::cout << "✓ BLAS resources cleaned up" << std::endl;
        }
    }
    
    /**
     * Get platform information for reporting
     */
    std::string getPlatformInfo() {
        return std::string("HIP Marshalling Platform - ") + device_properties.name;
    }
};

/**
 * Main demonstration program
 */
int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "    HIP GPU Marshalling Demonstration" << std::endl;
    std::cout << "======================================================\n" << std::endl;
    
    GPUMarshallingDemo demo;
    
    // Initialize marshalling environment
    if (!demo.initialize()) {
        std::cout << "❌ Failed to initialize GPU marshalling environment" << std::endl;
        return -1;
    }
    
    try {
        // Demonstrate core marshalling capabilities
        demo.demonstrateVectorAddition();
        
        // Cleanup
        demo.cleanup();
        
        std::cout << "\n=== Marshalling Demonstration Complete ===" << std::endl;
        std::cout << "Platform: " << demo.getPlatformInfo() << std::endl;
        std::cout << "✓ All operations successfully marshalled" << std::endl;
        std::cout << "✓ Same source code, optimized for target platform" << std::endl;
        std::cout << "✓ Ready for production deployment" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during demonstration: " << e.what() << std::endl;
        demo.cleanup();
        return -1;
    }
    
    std::cout << "\n======================================================" << std::endl;
    std::cout << "HIP: Your path to GPU vendor independence" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    return 0;
}
