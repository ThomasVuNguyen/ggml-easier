#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <chrono>
#include <string>

/**
 * This example demonstrates how to perform matrix multiplication using ggml-easy.h
 * 
 * Given 2 matrices A and B, the result matrix C is calculated as follows:
 *   C = (A x B) * 2
 *
 * We will use utils.debug_print() to debug the intermediate result of (A x B)
 * Then, we will use utils.mark_output() to get the final result of C
 *
 * The final result can be printed using ggml_easy::debug::print_tensor_data()
 * Or, can be used to perform further computations
 * 
 * Usage: ./basic-loop [num_iterations]
 * If num_iterations is not provided, it defaults to 100
 */

int main(int argc, char** argv) {
    // Parse command line arguments
    int num_iterations = 100; // Default value
    if (argc > 1) {
        try {
            num_iterations = std::stoi(argv[1]);
            if (num_iterations <= 0) {
                std::cerr << "Number of iterations must be positive. Using default (100)." << std::endl;
                num_iterations = 100;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid number of iterations. Using default (100)." << std::endl;
        }
    }

    // Setup logging levels
    ggml_log_level error_level = GGML_LOG_LEVEL_ERROR;
    ggml_log_level info_level = GGML_LOG_LEVEL_INFO;

    ggml_easy::ctx_params params;
    params.log_level = error_level; // Silence debug output
    ggml_easy::ctx ctx(params);

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // Optional: print backend buffer info once at the beginning
    ggml_easy::debug::print_backend_buffer_info(ctx);

    std::cout << "Running computation " << num_iterations << " times..." << std::endl;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Compute multiple times
    for (int i = 0; i < num_iterations; i++) {
        // Rebuild the computation graph for each iteration
        ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
            ggml_tensor * a = utils.new_input("a", GGML_TYPE_F32, cols_A, rows_A);
            ggml_tensor * b = utils.new_input("b", GGML_TYPE_F32, cols_B, rows_B);
            ggml_tensor * a_mul_b = ggml_mul_mat(ctx_gf, a, b);
            // Don't use debug_print in the loop
            ggml_tensor * result = ggml_scale(ctx_gf, a_mul_b, 2);
            utils.mark_output(result, "result");
        });

        // Set data for each iteration
        ctx.set_tensor_data("a", matrix_A);
        ctx.set_tensor_data("b", matrix_B);

        // Compute
        ggml_status status = ctx.compute();
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "error: ggml compute return status: " << status << std::endl;
            return 1;
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Completed " << num_iterations << " iterations in " << duration << " ms" << std::endl;
    std::cout << "Average time per iteration: " << (float)duration / num_iterations << " ms" << std::endl;

    // Print the final result of last iteration
    auto result = ctx.get_tensor_data("result");
    ggml_tensor * result_tensor = result.first;
    std::vector<uint8_t> & result_data = result.second;

    // Print result
    std::cout << "\nFinal result:" << std::endl;
    // Reset log level to INFO for the final output
    ggml_log_set(ggml_easy::log_cb, &info_level);
    ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    return 0;
}
