#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

void printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):" << std::endl;
    
    // Limit display to 8x8 for readability
    int display_rows = std::min(rows, 8);
    int display_cols = std::min(cols, 8);
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            std::cout << std::setw(8) << std::setprecision(2) << std::fixed 
                      << matrix[i * cols + j] << " ";
        }
        if (cols > 8) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 8) {
        std::cout << "..." << std::endl;
    }
}

void generateRandomMatrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    int n, m, k;
    
    std::cout << "=== rocBLAS Matrix Multiplication ===" << std::endl;
    std::cout << "This program multiplies two random matrices A * B = C" << std::endl;
    std::cout << "Matrix A will be n x m" << std::endl;
    std::cout << "Matrix B will be m x k" << std::endl;
    std::cout << "Result C will be n x k" << std::endl << std::endl;
    
    // Get matrix dimensions from user
    if (argc == 4) {
        // Command line arguments: program n m k
        n = std::atoi(argv[1]);
        m = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else {
        // Interactive input
        std::cout << "Enter dimensions:" << std::endl;
        std::cout << "n (rows of A, rows of result): ";
        std::cin >> n;
        std::cout << "m (cols of A, rows of B): ";
        std::cin >> m;
        std::cout << "k (cols of B, cols of result): ";
        std::cin >> k;
    }
    
    // Validate input
    if (n <= 0 || m <= 0 || k <= 0) {
        std::cerr << "Error: All dimensions must be positive!" << std::endl;
        return 1;
    }
    
    if (n > 10000 || m > 10000 || k > 10000) {
        std::cerr << "Error: Dimensions too large (max 10000)!" << std::endl;
        return 1;
    }
    
    std::cout << "\nMatrix dimensions:" << std::endl;
    std::cout << "A: " << n << " x " << m << std::endl;
    std::cout << "B: " << m << " x " << k << std::endl;
    std::cout << "C: " << n << " x " << k << std::endl;
    
    // Calculate matrix sizes
    int size_a = n * m;
    int size_b = m * k;
    int size_c = n * k;
    
    // Initialize rocBLAS
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    std::cout << "\nrocBLAS initialized successfully!" << std::endl;
    
    // Allocate host memory
    std::vector<float> h_a(size_a);
    std::vector<float> h_b(size_b);
    std::vector<float> h_c(size_c, 0.0f);
    
    // Generate random matrices
    std::cout << "Generating random matrices..." << std::endl;
    generateRandomMatrix(h_a, size_a);
    generateRandomMatrix(h_b, size_b);
    
    // Print matrices (if small enough)
    if (n <= 8 && m <= 8 && k <= 8) {
        printMatrix(h_a, n, m, "Matrix A");
        printMatrix(h_b, m, k, "Matrix B");
    } else {
        std::cout << "\nMatrices are large - showing first few elements:" << std::endl;
        std::cout << "A[0][0] = " << h_a[0] << ", A[0][1] = " << (m > 1 ? h_a[1] : 0) << std::endl;
        std::cout << "B[0][0] = " << h_b[0] << ", B[0][1] = " << (k > 1 ? h_b[1] : 0) << std::endl;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, size_a * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, size_b * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, size_c * sizeof(float)));
    
    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_a * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size_b * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, h_c.data(), size_c * sizeof(float), hipMemcpyHostToDevice));
    
    std::cout << "\nPerforming matrix multiplication on GPU..." << std::endl;
    
    // Perform matrix multiplication: C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // Note: rocBLAS uses column-major ordering, so we need to adjust parameters
    // For our row-major C = A * B, we compute C^T = B^T * A^T
    ROCBLAS_CHECK(rocblas_sgemm(handle,
                               rocblas_operation_none,    // Don't transpose B
                               rocblas_operation_none,    // Don't transpose A
                               k,                         // Number of rows of B^T and C^T
                               n,                         // Number of columns of A^T and C^T  
                               m,                         // Number of columns of B^T and rows of A^T
                               &alpha,                    // alpha scalar
                               d_b,                       // Matrix B (treated as B^T)
                               k,                         // Leading dimension of B
                               d_a,                       // Matrix A (treated as A^T)
                               m,                         // Leading dimension of A
                               &beta,                     // beta scalar
                               d_c,                       // Matrix C (result, treated as C^T)
                               k));                       // Leading dimension of C
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, size_c * sizeof(float), hipMemcpyDeviceToHost));
    
    std::cout << "Matrix multiplication completed!" << std::endl;
    
    // Print result matrix
    if (n <= 8 && k <= 8) {
        printMatrix(h_c, n, k, "Result C = A * B");
    } else {
        std::cout << "\nResult matrix C (" << n << "x" << k << ") - showing first few elements:" << std::endl;
        for (int i = 0; i < std::min(n, 4); i++) {
            for (int j = 0; j < std::min(k, 4); j++) {
                std::cout << std::setw(8) << std::setprecision(2) << std::fixed 
                          << h_c[i * k + j] << " ";
            }
            if (k > 4) std::cout << "...";
            std::cout << std::endl;
        }
        if (n > 4) std::cout << "..." << std::endl;
    }
    
    // Optional: Verify result with CPU calculation (for small matrices)
    if (n <= 10 && m <= 10 && k <= 10) {
        std::cout << "\nVerifying result with CPU calculation..." << std::endl;
        std::vector<float> cpu_result(size_c, 0.0f);
        
        // CPU matrix multiplication
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                for (int l = 0; l < m; l++) {
                    cpu_result[i * k + j] += h_a[i * m + l] * h_b[l * k + j];
                }
            }
        }
        
        // Compare results
        bool match = true;
        float max_error = 0.0f;
        float tolerance = 1e-4f;
        
        for (int i = 0; i < size_c; i++) {
            float error = std::abs(h_c[i] - cpu_result[i]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                match = false;
            }
        }
        
        if (match) {
            std::cout << "✓ Verification PASSED! (max error: " << max_error << ")" << std::endl;
        } else {
            std::cout << "✗ Verification FAILED! (max error: " << max_error << ")" << std::endl;
        }
    }
    
    // Calculate and display performance info
    long long total_ops = 2LL * n * m * k;  // Each element of C requires m multiply-adds
    std::cout << "\nPerformance Info:" << std::endl;
    std::cout << "Total operations: " << total_ops << " FLOPS" << std::endl;
    std::cout << "Matrix A size: " << (size_a * sizeof(float)) / 1024.0 << " KB" << std::endl;
    std::cout << "Matrix B size: " << (size_b * sizeof(float)) / 1024.0 << " KB" << std::endl;
    std::cout << "Matrix C size: " << (size_c * sizeof(float)) / 1024.0 << " KB" << std::endl;
    
    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    rocblas_destroy_handle(handle);
    
    std::cout << "\nProgram completed successfully!" << std::endl;
    
    return 0;
}
