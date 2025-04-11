#include "ggml-easy.h"
#include <iostream>
#include <random>
// List of common weight tensor names to try
/* const char* tensor_names[] = {
    "token_embd.weight",            // Token embeddings
    "output.weight",                // Output layer
    "blk.0.attn_q.weight",          // First layer query weights
    "blk.0.attn_k.weight",          // First layer key weights 
    "blk.0.attn_v.weight",          // First layer value weights
    "blk.0.ffn_gate.weight",        // First layer FFN gate
    "blk.0.ffn_up.weight",          // First layer FFN up projection
    "blk.0.ffn_down.weight",        // First layer FFN down projection
    "blk.0.attn_norm.weight",       // First layer attention norm
    "blk.0.ffn_norm.weight",        // First layer FFN norm
    "norm.weight"                   // Final norm
};*/
int read_gguf_model(const char* model_path, const char* weight_name){
    ggml_easy::ctx_params params;
    params.use_gpu = false; // I just don't like GPU
    ggml_easy::ctx* ctx = new ggml_easy::ctx(params);
    try {
        // Load the GGUF file
        ctx->load_gguf(model_path);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception& e){
        printf("Error loading gguf\n");
    }

    try {
        ggml_tensor* tensor = ctx->get_weight(weight_name);
        std::cout << "\nFound tensor: " << weight_name << std::endl;
        std::cout << "  Shape: [";
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << tensor->ne[i];
        }
        std::cout << "]" << std::endl;
        
        // Print some random values from the tensor
        std::random_device rd;
        std::mt19937 gen(rd());
        
        float* data = (float*)tensor->data;
        int64_t n_elements = ggml_nelements(tensor);
        std::cout << "  Sample values: ";
        
        // Print up to 5 random values
        for (int i = 0; i < 5 && i < n_elements; i++) {
            std::uniform_int_distribution<> distrib(0, n_elements - 1);
            int idx = distrib(gen);
            std::cout << data[idx] << " ";
        }
        std::cout << std::endl;
        // Print all values from the tensor
        std::cout << "  All values: ";
        for (int64_t i = 0; i < n_elements; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        // This tensor name might not exist in this model
        printf("No %s weight found.", weight_name);
    }

    delete ctx;

    return 1;
}


int main(int argc, char** argv) {
    int result = read_gguf_model("llama3.2-gguf/Llama-3.2-1B.Q8_0.gguf", "blk.0.ffn_down.weight");
    return 0;
}
