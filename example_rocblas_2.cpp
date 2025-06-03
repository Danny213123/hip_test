#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>

// Matrix dimensions
const int MATRIX_SIZE = 10;
const int MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__ << " - status: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

class MatrixBenchmark {
private:
    rocblas_handle handle;
    std::vector<float*> d_matrices_a;  // Device matrices A
    std::vector<float*> d_matrices_b;  // Device matrices B  
    std::vector<float*> d_matrices_c;  // Device matrices C (results)
    std::vector<std::vector<float>> h_matrices_a;  // Host matrices A
    std::vector<std::vector<float>> h_matrices_b;  // Host matrices B
    std::vector<std::vector<float>> h_matrices_c;  // Host matrices C
    int num_matrices;
    
public:
    MatrixBenchmark(int n) : num_matrices(n) {
        // Initialize rocBLAS
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
        
        // Set rocBLAS to use GPU pointers
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        
        std::cout << "rocBLAS initialized successfully!" << std::endl;
        
        // Initialize matrices
        initializeMatrices();
    }
    
    ~MatrixBenchmark() {
        cleanup();
        rocblas_destroy_handle(handle);
    }
    
    void initializeMatrices() {
        std::cout << "Initializing " << num_matrices << " pairs of 10x10 matrices..." << std::endl;
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
        
        // Allocate host memory
        h_matrices_a.resize(num_matrices, std::vector<float>(MATRIX_ELEMENTS));
        h_matrices_b.resize(num_matrices, std::vector<float>(MATRIX_ELEMENTS));
        h_matrices_c.resize(num_matrices, std::vector<float>(MATRIX_ELEMENTS));
        
        // Generate random matrices
        for (int i = 0; i < num_matrices; i++) {
            for (int j = 0; j < MATRIX_ELEMENTS; j++) {
                h_matrices_a[i][j] = dis(gen);
                h_matrices_b[i][j] = dis(gen);
                h_matrices_c[i][j] = 0.0f;  // Initialize result to zero
            }
        }
        
        // Allocate device memory
        d_matrices_a.resize(num_matrices);
        d_matrices_b.resize(num_matrices);
        d_matrices_c.resize(num_matrices);
        
        for (int i = 0; i < num_matrices; i++) {
            HIP_CHECK(hipMalloc(&d_matrices_a[i], MATRIX_ELEMENTS * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_matrices_b[i], MATRIX_ELEMENTS * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_matrices_c[i], MATRIX_ELEMENTS * sizeof(float)));
            
            // Copy data to device
            HIP_CHECK(hipMemcpy(d_matrices_a[i], h_matrices_a[i].data(), 
                               MATRIX_ELEMENTS * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_matrices_b[i], h_matrices_b[i].data(), 
                               MATRIX_ELEMENTS * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_matrices_c[i], h_matrices_c[i].data(), 
                               MATRIX_ELEMENTS * sizeof(float), hipMemcpyHostToDevice));
        }
        
        std::cout << "Matrices initialized and copied to GPU!" << std::endl;
    }
    
    void runBenchmark(int iterations) {
        std::cout << "\n=== Starting Benchmark ===" << std::endl;
        std::cout << "Matrices: " << num_matrices << " pairs of 10x10" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Total operations: " << num_matrices * iterations << std::endl;
        
        // Warm up
        std::cout << "Warming up..." << std::endl;
        performMatrixMultiplications();
        HIP_CHECK(hipDeviceSynchronize());
        
        // Benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < iterations; iter++) {
            performMatrixMultiplications();
        }
        
        // Ensure all operations complete
        HIP_CHECK(hipDeviceSynchronize());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Calculate performance metrics
        double total_time_ms = duration.count() / 1000.0;
        double total_operations = num_matrices * iterations;
        double operations_per_second = total_operations / (total_time_ms / 1000.0);
        double gflops = calculateGFLOPS(total_operations, total_time_ms / 1000.0);
        
        // Print results
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
        std::cout << "Average time per iteration: " << total_time_ms / iterations << " ms" << std::endl;
        std::cout << "Average time per matrix multiplication: " << total_time_ms / total_operations << " ms" << std::endl;
        std::cout << "Matrix multiplications per second: " << std::setprecision(0) << operations_per_second << std::endl;
        std::cout << std::setprecision(3) << "Performance: " << gflops << " GFLOPS" << std::endl;
        
        // Verify results (optional)
        verifyResults();
    }
    
private:
    void performMatrixMultiplications() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        for (int i = 0; i < num_matrices; i++) {
            // Perform C = alpha * A * B + beta * C
            // rocblas_sgemm parameters: handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                                       rocblas_operation_none,     // No transpose on A
                                       rocblas_operation_none,     // No transpose on B
                                       MATRIX_SIZE,                // m (rows of A and C)
                                       MATRIX_SIZE,                // n (columns of B and C)
                                       MATRIX_SIZE,                // k (columns of A, rows of B)
                                       &alpha,                     // alpha scalar
                                       d_matrices_a[i],           // Matrix A
                                       MATRIX_SIZE,               // Leading dimension of A
                                       d_matrices_b[i],           // Matrix B
                                       MATRIX_SIZE,               // Leading dimension of B
                                       &beta,                     // beta scalar
                                       d_matrices_c[i],           // Matrix C (output)
                                       MATRIX_SIZE));             // Leading dimension of C
        }
    }
    
    double calculateGFLOPS(double operations, double time_seconds) {
        // Each matrix multiplication is 2*n^3 FLOPs (for n x n matrices)
        double flops_per_multiplication = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
        double total_flops = operations * flops_per_multiplication;
        return total_flops / (time_seconds * 1e9);
    }
    
    void verifyResults() {
        std::cout << "\nVerifying first matrix result..." << std::endl;
        
        // Copy result back to host for verification
        std::vector<float> result(MATRIX_ELEMENTS);
        HIP_CHECK(hipMemcpy(result.data(), d_matrices_c[0], 
                           MATRIX_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate expected result on CPU for first matrix
        std::vector<float> expected(MATRIX_ELEMENTS, 0.0f);
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    expected[i * MATRIX_SIZE + j] += 
                        h_matrices_a[0][i * MATRIX_SIZE + k] * h_matrices_b[0][k * MATRIX_SIZE + j];
                }
            }
        }
        
        // Check if results match (within tolerance)
        bool verification_passed = true;
        float tolerance = 1e-4f;
        float max_error = 0.0f;
        
        for (int i = 0; i < MATRIX_ELEMENTS; i++) {
            float error = std::abs(result[i] - expected[i]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                verification_passed = false;
            }
        }
        
        if (verification_passed) {
            std::cout << "✓ Verification PASSED (max error: " << max_error << ")" << std::endl;
        } else {
            std::cout << "✗ Verification FAILED (max error: " << max_error << ")" << std::endl;
        }
        
        // Print sample of results
        std::cout << "\nSample results (first 3x3 of first matrix):" << std::endl;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << std::setw(8) << std::setprecision(2) << result[i * MATRIX_SIZE + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    void cleanup() {
        for (int i = 0; i < num_matrices; i++) {
            if (d_matrices_a[i]) hipFree(d_matrices_a[i]);
            if (d_matrices_b[i]) hipFree(d_matrices_b[i]);
            if (d_matrices_c[i]) hipFree(d_matrices_c[i]);
        }
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n <number>    Number of matrix pairs to multiply (default: 100)" << std::endl;
    std::cout << "  -m <number>    Number of iterations (default: 10)" << std::endl;
    std::cout << "  -h, --help     Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example: " << program_name << " -n 500 -m 20" << std::endl;
}

void printSystemInfo() {
    std::cout << "=== System Information ===" << std::endl;
    
    // GPU information
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "Number of GPU devices: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::cout << "GPU 0: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
        if (strlen(prop.gcnArchName) > 0) {
            std::cout << "Architecture: " << prop.gcnArchName << std::endl;
        }
    }
    
    // rocBLAS version (if available)
    char version[256];
    size_t size = sizeof(version);
    if (rocblas_get_version_string(version, size) == rocblas_status_success) {
        std::cout << "rocBLAS Version: " << version << std::endl;
    }
    
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    int num_matrices = 100;  // Default: 100 matrix pairs
    int iterations = 10;     // Default: 10 iterations
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_matrices = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate arguments
    if (num_matrices <= 0 || iterations <= 0) {
        std::cerr << "Error: Number of matrices and iterations must be positive" << std::endl;
        return 1;
    }
    
    std::cout << "rocBLAS Matrix Multiplication Benchmark" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        printSystemInfo();
        
        // Create and run benchmark
        MatrixBenchmark benchmark(num_matrices);
        benchmark.runBenchmark(iterations);
        
        std::cout << "\nBenchmark completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
