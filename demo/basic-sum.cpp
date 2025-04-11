#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

int main(){
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    const int rows_X = 4, cols_X = 2;
    float matrix_X[rows_X * cols_X] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    const int rows_Y = 4, cols_Y = 2;
    float matrix_Y[rows_Y * cols_Y] = {
        9, 8,
        7, 6,
        5, 4,
        3, 2,
    };

    // Note: be careful with the braces
    ctx.build_graph([&](ggml_context* ctx_gf, ggml_cgraph* gf, auto & utils){
        ggml_tensor* x = utils.new_input("x", GGML_TYPE_F32, cols_X, rows_X);
        ggml_tensor* y = utils.new_input("y", GGML_TYPE_F32, cols_Y, rows_Y);
        ggml_tensor* a_add_b = ggml_add(ctx_gf, x, y);
        utils.debug_print(a_add_b, "a_add_b");
        ggml_tensor* result = ggml_scale(ctx_gf, a_add_b, 2);
        utils.mark_output(result, "result");
    });

    ctx.set_tensor_data("x", matrix_X);
    ctx.set_tensor_data("y", matrix_Y);

    // print backend buffer info
    // cuz we can
    ggml_easy::debug::print_backend_buffer_info(ctx);

    // let her go brr
    ggml_status status = ctx.compute();
    if (status != GGML_STATUS_SUCCESS){
        std::cerr << "Shit fucked up: " << status << std::endl;
        return 1;
    }

    // get result
    auto result = ctx.get_tensor_data("result");
    ggml_tensor* result_tensor = result.first;
    std::vector<uint8_t> & result_data = result.second;

    // print out result
    ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    // Just a good practice to notify that program ran without error
    return 0;
};