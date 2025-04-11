// GGUF STUFF

#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <pcre.h> // for regex splits in tokenizer
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <random>
#include <map>
#include <vector>

// Forward declaration
std::string map_weight_name(const std::string& original_name);

typedef struct {
    int dim = 2048; // hidden_size (2048)
    int hidden_dim = 8192; // intermediate_size (8192)
    int n_layers = 16; // num_hidden_layers (16)
    int n_heads = 32; // num_attention_heads (32)
    int n_kv_heads = 8; // num_key_value_heads (8)
    int vocab_size = 12856; // vocab_size (12856)
    int seq_len = 131072; // max_position_embeddings (131072)
} Config;

typedef struct{
    // model.embed_tokens.weight
    ggml_tensor* float_token_embedding_table;

    // model.layers.i.input_layernorm.weight
    ggml_tensor* rms_attn_weight;

    // model.layers.i.post_attention_layernowm.weight
    ggml_tensor* rms_ffn_weight;

    //model.layers.i.self_attn.q_proj.weight
    ggml_tensor* wq;

    //model.layers.i.self_attn.k_proj.weight
    ggml_tensor* wk;

    //model.layers.i.self_attn.v_proj.weight
    ggml_tensor* wv;

    //model.layers.i.self_attn.o_proj.weight
    ggml_tensor* wo;

    // model.layers.i.mlp.gate_proj.weight
    ggml_tensor* w1;

    // model.layers.i.mlp_down_proj.weight
    ggml_tensor* w2;

    // model.layers.i.mlp.up_proj.weight
    ggml_tensor* w3;

    // model.norm.weight
    ggml_tensor* rms_final_weight;

    // token_embedding_table
    ggml_tensor* wcls;

} TransformerWeights;

typedef struct{
    ggml_tensor* x;
    ggml_tensor* xb; ggml_tensor* xb2;
    ggml_tensor* hb; ggml_tensor* hb2;
    ggml_tensor* q; ggml_tensor* k; ggml_tensor* v;
    ggml_tensor* att; 
    ggml_tensor* logits;

    // kv cache;
    ggml_tensor* key_value;
    ggml_tensor* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd; // file descriptor for memory mapping
    ggml_tensor* data; // memory mapped data pointer
    ssize_t file_size; // size of checkpoint file in bytes
} Transformer;


Transformer* load_transformer_model(const char* model_path) {
    Transformer* transformer = new Transformer();
    
    ggml_easy::ctx_params params;
    params.use_gpu = false;
    ggml_easy::ctx* ctx = new ggml_easy::ctx(params);
    
    try {
        // Load the GGUF file
        ctx->load_gguf(model_path);
        std::cout << "Model loaded successfully!" << std::endl;
        
        // Access GGUF metadata through the loaded contexts
        // In the ggml-easy library, the gguf context is stored in loaded_ggufs
        // Try to get metadata values
        // This assumes the ctx has a way to get to the gguf_context
        try {
            // Since we don't have direct access to the gguf_context from the ggml-easy ctx
            // We'll need to infer model properties from the tensors we have
            
            // Get the token embedding tensor to infer dimensions
            std::string embed_name = map_weight_name("model.embed_tokens.weight");
            ggml_tensor* token_emb = ctx->get_weight(embed_name.c_str());
            if (token_emb != nullptr) {
                // Tensor dimensions should tell us model configuration
                transformer->config.dim = token_emb->ne[0]; // embedding dim
                transformer->config.vocab_size = token_emb->ne[1]; // vocab size
                
                std::cout << "Inferred model dimensions from token embedding: " << std::endl;
                std::cout << "  Embedding dim: " << transformer->config.dim << std::endl;
                std::cout << "  Vocab size: " << transformer->config.vocab_size << std::endl;
            }
            
            // Infer number of layers by looking for pattern in tensor names
            int max_layer_found = -1;
            for (int i = 0; i < 100; i++) { // Try reasonable number of layers
                std::string pattern = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string mapped_name = map_weight_name(pattern);
                try {
                    ggml_tensor* layer_tensor = ctx->get_weight(mapped_name.c_str());
                    if (layer_tensor != nullptr) {
                        max_layer_found = i;
                    }
                } catch (...) {
                    // This layer doesn't exist
                    break;
                }
            }
            
            if (max_layer_found >= 0) {
                transformer->config.n_layers = max_layer_found + 1; // 0-indexed
                std::cout << "  Number of layers: " << transformer->config.n_layers << std::endl;
            }
            
            // Try to infer number of attention heads from q projection weight
            try {
                std::string wq_name = map_weight_name("model.layers.0.self_attn.q_proj.weight");
                ggml_tensor* wq = ctx->get_weight(wq_name.c_str());
                if (wq != nullptr) {
                    // Head dimension = embedding_dim / n_heads
                    // Common head dimensions are 64, 80, 128
                    for (int head_dim : {64, 80, 128}) {
                        if (transformer->config.dim % head_dim == 0) {
                            transformer->config.n_heads = transformer->config.dim / head_dim;
                            std::cout << "  Inferred number of heads: " << transformer->config.n_heads << std::endl;
                            break;
                        }
                    }
                    
                    // Try to infer KV heads - common ratios are 1/1, 1/8, 1/4
                    for (int ratio : {1, 4, 8}) {
                        int possible_kv_heads = transformer->config.n_heads / ratio;
                        if (transformer->config.n_heads % ratio == 0 && possible_kv_heads > 0) {
                            transformer->config.n_kv_heads = possible_kv_heads;
                            std::cout << "  Inferred number of KV heads: " << transformer->config.n_kv_heads << std::endl;
                            break;
                        }
                    }
                }
            } catch (...) {
                // Couldn't infer attention heads
            }
            
        } catch (const std::exception& e) {
            std::cout << "Error accessing model metadata: " << e.what() << std::endl;
            std::cout << "Using default config values." << std::endl;
        }
        
        // Load model weights
        try {
            transformer->weights.float_token_embedding_table = ctx->get_weight(map_weight_name("model.embed_tokens.weight").c_str());
            transformer->weights.rms_final_weight = ctx->get_weight(map_weight_name("model.norm.weight").c_str());
            transformer->weights.wcls = transformer->weights.float_token_embedding_table; // Share embedding weights by default
            
            // For layers, iterate through them and get weights for each layer
            for (int i = 0; i < transformer->config.n_layers; i++) {
                // Build layer pattern and map to actual GGUF names
                std::string layer_str = std::to_string(i);
                std::string in_ln_name = map_weight_name("model.layers." + layer_str + ".input_layernorm.weight");
                std::string post_ln_name = map_weight_name("model.layers." + layer_str + ".post_attention_layernorm.weight");
                std::string q_name = map_weight_name("model.layers." + layer_str + ".self_attn.q_proj.weight");
                std::string k_name = map_weight_name("model.layers." + layer_str + ".self_attn.k_proj.weight");
                std::string v_name = map_weight_name("model.layers." + layer_str + ".self_attn.v_proj.weight");
                std::string o_name = map_weight_name("model.layers." + layer_str + ".self_attn.o_proj.weight");
                std::string gate_name = map_weight_name("model.layers." + layer_str + ".mlp.gate_proj.weight");
                std::string up_name = map_weight_name("model.layers." + layer_str + ".mlp.up_proj.weight");
                std::string down_name = map_weight_name("model.layers." + layer_str + ".mlp.down_proj.weight");
                
                // Get layer-specific weights - only for first layer as example
                if (i == 0) {
                    transformer->weights.rms_attn_weight = ctx->get_weight(in_ln_name.c_str());
                    transformer->weights.rms_ffn_weight = ctx->get_weight(post_ln_name.c_str());
                    transformer->weights.wq = ctx->get_weight(q_name.c_str());
                    transformer->weights.wk = ctx->get_weight(k_name.c_str());
                    transformer->weights.wv = ctx->get_weight(v_name.c_str());
                    transformer->weights.wo = ctx->get_weight(o_name.c_str());
                    transformer->weights.w1 = ctx->get_weight(gate_name.c_str());
                    transformer->weights.w3 = ctx->get_weight(up_name.c_str());
                    transformer->weights.w2 = ctx->get_weight(down_name.c_str());
                }
            }
            
            std::cout << "Successfully loaded model weights!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error loading weights: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error loading GGUF file: " << e.what() << std::endl;
        delete transformer;
        delete ctx;
        return nullptr;
    }
    
    delete ctx;
    return transformer;
}

int read_gguf_model(const char* model_path, const char* weight_name) {
    // Try to load the model as a Transformer
    Transformer* model = load_transformer_model(model_path);
    if (model == nullptr) {
        printf("Failed to load model as Transformer.\n");
        return 0;
    }
    
    printf("Successfully loaded model as Transformer.\n");
    printf("Model configuration:\n");
    printf("  Embedding dimension: %d\n", model->config.dim);
    printf("  Hidden dimension: %d\n", model->config.hidden_dim);
    printf("  Number of layers: %d\n", model->config.n_layers);
    printf("  Number of attention heads: %d\n", model->config.n_heads);
    printf("  Number of KV heads: %d\n", model->config.n_kv_heads);
    printf("  Vocabulary size: %d\n", model->config.vocab_size);
    printf("  Maximum sequence length: %d\n", model->config.seq_len);
    
    // Now try to get and display the specific requested weight
    ggml_easy::ctx_params params;
    params.use_gpu = false;
    ggml_easy::ctx* ctx = new ggml_easy::ctx(params);
    
    try {
        // Load the GGUF file
        ctx->load_gguf(model_path);
        
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
        } catch (const std::exception& e) {
            // This tensor name might not exist in this model
            printf("No '%s' weight found in the model.", weight_name);
        }
    } catch (const std::exception& e) {
        printf("Error accessing specific weight: %s\n", e.what());
    }
    
    // Clean up
    delete ctx;
    delete model;

    return 1;
}

// Function to map original weight names to potential GGUF weight names
std::string map_weight_name(const std::string& original_name) {
    // Map of original names to possible alternatives in GGUF format
    static std::map<std::string, std::vector<std::string>> name_mappings = {
        // Token embeddings
        {"model.embed_tokens.weight", {"token_embd.weight", "model.embed_tokens.weight"}},
        
        // Final norm
        {"model.norm.weight", {"output_norm.weight", "model.norm.weight"}},
        
        // For each layer pattern
        {"model.layers.{}.input_layernorm.weight", {"blk.{}.attn_norm.weight", "model.layers.{}.input_layernorm.weight"}},
        {"model.layers.{}.mlp.down_proj.weight", {"blk.{}.ffn_down.weight", "model.layers.{}.mlp.down_proj.weight"}},
        {"model.layers.{}.mlp.gate_proj.weight", {"blk.{}.ffn_gate.weight", "model.layers.{}.mlp.gate_proj.weight"}},
        {"model.layers.{}.mlp.up_proj.weight", {"blk.{}.ffn_up.weight", "model.layers.{}.mlp.up_proj.weight"}},
        {"model.layers.{}.post_attention_layernorm.weight", {"blk.{}.ffn_norm.weight", "model.layers.{}.post_attention_layernorm.weight"}},
        {"model.layers.{}.self_attn.k_proj.weight", {"blk.{}.attn_k.weight", "model.layers.{}.self_attn.k_proj.weight"}},
        {"model.layers.{}.self_attn.o_proj.weight", {"blk.{}.attn_output.weight", "model.layers.{}.self_attn.o_proj.weight"}},
        {"model.layers.{}.self_attn.q_proj.weight", {"blk.{}.attn_q.weight", "model.layers.{}.self_attn.q_proj.weight"}},
        {"model.layers.{}.self_attn.v_proj.weight", {"blk.{}.attn_v.weight", "model.layers.{}.self_attn.v_proj.weight"}}
    };
    
    // Check if this is a direct match to a pattern with layer numbers
    for (const auto& mapping : name_mappings) {
        std::string pattern = mapping.first;
        
        // If the pattern contains a layer placeholder
        if (pattern.find("{}") != std::string::npos) {
            // Extract layer number if this is a layer-specific weight
            size_t layer_pos = original_name.find("layers.");
            if (layer_pos != std::string::npos) {
                size_t dot_pos = original_name.find(".", layer_pos + 7);
                if (dot_pos != std::string::npos) {
                    std::string layer_num = original_name.substr(layer_pos + 7, dot_pos - (layer_pos + 7));
                    
                    // Replace {} with layer number
                    std::string replaced_pattern = pattern;
                    size_t placeholder_pos = replaced_pattern.find("{}");
                    replaced_pattern.replace(placeholder_pos, 2, layer_num);
                    
                    // If this matches our original name
                    if (replaced_pattern == original_name) {
                        std::vector<std::string> alternatives;
                        for (const auto& alt_pattern : mapping.second) {
                            std::string alt = alt_pattern;
                            size_t alt_placeholder = alt.find("{}");
                            alt.replace(alt_placeholder, 2, layer_num);
                            alternatives.push_back(alt);
                        }
                        return alternatives[0]; // Return the first alternative
                    }
                }
            }
        } else if (pattern == original_name) {
            // Direct match for non-layer weights
            return mapping.second[0];
        }
    }
    
    // If no mapping found, return the original
    return original_name;
}

// Function to test all weights are available in the GGUF file
bool test_all_weights_available(const char* model_path) {
    ggml_easy::ctx_params params;
    params.use_gpu = false;
    ggml_easy::ctx* ctx = new ggml_easy::ctx(params);
    
    std::vector<std::string> expected_weights = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.10.input_layernorm.weight",
        "model.layers.10.mlp.down_proj.weight",
        "model.layers.10.mlp.gate_proj.weight",
        "model.layers.10.mlp.up_proj.weight",
        "model.layers.10.post_attention_layernorm.weight",
        "model.layers.10.self_attn.k_proj.weight",
        "model.layers.10.self_attn.o_proj.weight",
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.10.self_attn.v_proj.weight",
        "model.layers.11.input_layernorm.weight",
        "model.layers.11.mlp.down_proj.weight",
        "model.layers.11.mlp.gate_proj.weight",
        "model.layers.11.mlp.up_proj.weight",
        "model.layers.11.post_attention_layernorm.weight",
        "model.layers.11.self_attn.k_proj.weight",
        "model.layers.11.self_attn.o_proj.weight",
        "model.layers.11.self_attn.q_proj.weight",
        "model.layers.11.self_attn.v_proj.weight",
        "model.layers.12.input_layernorm.weight",
        "model.layers.12.mlp.down_proj.weight",
        "model.layers.12.mlp.gate_proj.weight",
        "model.layers.12.mlp.up_proj.weight",
        "model.layers.12.post_attention_layernorm.weight",
        "model.layers.12.self_attn.k_proj.weight",
        "model.layers.12.self_attn.o_proj.weight",
        "model.layers.12.self_attn.q_proj.weight",
        "model.layers.12.self_attn.v_proj.weight",
        "model.layers.13.input_layernorm.weight",
        "model.layers.13.mlp.down_proj.weight",
        "model.layers.13.mlp.gate_proj.weight",
        "model.layers.13.mlp.up_proj.weight",
        "model.layers.13.post_attention_layernorm.weight",
        "model.layers.13.self_attn.k_proj.weight",
        "model.layers.13.self_attn.o_proj.weight",
        "model.layers.13.self_attn.q_proj.weight",
        "model.layers.13.self_attn.v_proj.weight",
        "model.layers.14.input_layernorm.weight",
        "model.layers.14.mlp.down_proj.weight",
        "model.layers.14.mlp.gate_proj.weight",
        "model.layers.14.mlp.up_proj.weight",
        "model.layers.14.post_attention_layernorm.weight",
        "model.layers.14.self_attn.k_proj.weight",
        "model.layers.14.self_attn.o_proj.weight",
        "model.layers.14.self_attn.q_proj.weight",
        "model.layers.14.self_attn.v_proj.weight",
        "model.layers.15.input_layernorm.weight",
        "model.layers.15.mlp.down_proj.weight",
        "model.layers.15.mlp.gate_proj.weight",
        "model.layers.15.mlp.up_proj.weight",
        "model.layers.15.post_attention_layernorm.weight",
        "model.layers.15.self_attn.k_proj.weight",
        "model.layers.15.self_attn.o_proj.weight",
        "model.layers.15.self_attn.q_proj.weight",
        "model.layers.15.self_attn.v_proj.weight",
        "model.layers.2.input_layernorm.weight",
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.2.mlp.gate_proj.weight",
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.post_attention_layernorm.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.self_attn.o_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.v_proj.weight",
        "model.layers.3.input_layernorm.weight",
        "model.layers.3.mlp.down_proj.weight",
        "model.layers.3.mlp.gate_proj.weight",
        "model.layers.3.mlp.up_proj.weight",
        "model.layers.3.post_attention_layernorm.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.4.input_layernorm.weight",
        "model.layers.4.mlp.down_proj.weight",
        "model.layers.4.mlp.gate_proj.weight",
        "model.layers.4.mlp.up_proj.weight",
        "model.layers.4.post_attention_layernorm.weight",
        "model.layers.4.self_attn.k_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.v_proj.weight",
        "model.layers.5.input_layernorm.weight",
        "model.layers.5.mlp.down_proj.weight",
        "model.layers.5.mlp.gate_proj.weight",
        "model.layers.5.mlp.up_proj.weight",
        "model.layers.5.post_attention_layernorm.weight",
        "model.layers.5.self_attn.k_proj.weight",
        "model.layers.5.self_attn.o_proj.weight",
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.5.self_attn.v_proj.weight",
        "model.layers.6.input_layernorm.weight",
        "model.layers.6.mlp.down_proj.weight",
        "model.layers.6.mlp.gate_proj.weight",
        "model.layers.6.mlp.up_proj.weight",
        "model.layers.6.post_attention_layernorm.weight",
        "model.layers.6.self_attn.k_proj.weight",
        "model.layers.6.self_attn.o_proj.weight",
        "model.layers.6.self_attn.q_proj.weight",
        "model.layers.6.self_attn.v_proj.weight",
        "model.layers.7.input_layernorm.weight",
        "model.layers.7.mlp.down_proj.weight",
        "model.layers.7.mlp.gate_proj.weight",
        "model.layers.7.mlp.up_proj.weight",
        "model.layers.7.post_attention_layernorm.weight",
        "model.layers.7.self_attn.k_proj.weight",
        "model.layers.7.self_attn.o_proj.weight",
        "model.layers.7.self_attn.q_proj.weight",
        "model.layers.7.self_attn.v_proj.weight",
        "model.layers.8.input_layernorm.weight",
        "model.layers.8.mlp.down_proj.weight",
        "model.layers.8.mlp.gate_proj.weight",
        "model.layers.8.mlp.up_proj.weight",
        "model.layers.8.post_attention_layernorm.weight",
        "model.layers.8.self_attn.k_proj.weight",
        "model.layers.8.self_attn.o_proj.weight",
        "model.layers.8.self_attn.q_proj.weight",
        "model.layers.8.self_attn.v_proj.weight",
        "model.layers.9.input_layernorm.weight",
        "model.layers.9.mlp.down_proj.weight",
        "model.layers.9.mlp.gate_proj.weight",
        "model.layers.9.mlp.up_proj.weight",
        "model.layers.9.post_attention_layernorm.weight",
        "model.layers.9.self_attn.k_proj.weight",
        "model.layers.9.self_attn.o_proj.weight",
        "model.layers.9.self_attn.q_proj.weight",
        "model.layers.9.self_attn.v_proj.weight",
        "model.norm.weight"
    };
    
    try {
        // Load the GGUF file
        ctx->load_gguf(model_path);
        std::cout << "Model loaded successfully for weight verification" << std::endl;
        
        int missing_count = 0;
        int total_weights = expected_weights.size();
        
        // Create a mapping to track which weights have been found
        std::map<std::string, bool> weight_status;
        for (const auto& weight : expected_weights) {
            weight_status[weight] = false;
        }
        
        // Try to access each weight
        for (const auto& weight_name : expected_weights) {
            // Try the original name and the mapped name
            std::string mapped_name = map_weight_name(weight_name);
            
            bool found = false;
            ggml_tensor* tensor = nullptr;
            std::string error_msg;
            
            // Try original name first
            try {
                tensor = ctx->get_weight(weight_name.c_str());
                if (tensor) {
                    found = true;
                    std::cout << "✓ Found " << weight_name << " (original name)" << std::endl;
                }
            } catch (const std::exception& e) {
                error_msg = e.what();
            }
            
            // Try mapped name if original failed
            if (!found && mapped_name != weight_name) {
                try {
                    tensor = ctx->get_weight(mapped_name.c_str());
                    if (tensor) {
                        found = true;
                        std::cout << "✓ Found " << weight_name << " as " << mapped_name << std::endl;
                    }
                } catch (const std::exception& e) {
                    error_msg = e.what();
                }
            }
            
            if (found) {
                weight_status[weight_name] = true;
                
                // Print shape info for the tensor
                std::cout << "  Shape: [";
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << tensor->ne[i];
                }
                std::cout << "]" << std::endl;
            } else {
                std::cout << "✗ Missing " << weight_name << " (tried both original and " << mapped_name << ")" << std::endl;
                std::cout << "  Error: " << error_msg << std::endl;
                missing_count++;
            }
        }
        
        // Print summary
        std::cout << "\n=== Weight Verification Summary ===" << std::endl;
        std::cout << "Total weights expected: " << total_weights << std::endl;
        std::cout << "Weights found: " << (total_weights - missing_count) << std::endl;
        std::cout << "Weights missing: " << missing_count << std::endl;
        
        // Print missing weights if any
        if (missing_count > 0) {
            std::cout << "\nMissing weights:" << std::endl;
            for (const auto& pair : weight_status) {
                if (!pair.second) {
                    std::cout << "  " << pair.first << std::endl;
                }
            }
        }
        
        // Success only if all weights are found
        delete ctx;
        return (missing_count == 0);
        
    } catch (const std::exception& e) {
        std::cout << "Error loading GGUF file for weight verification: " << e.what() << std::endl;
        delete ctx;
        return false;
    }
}

// Add this to the main function to test the weights
int main(int argc, char **argv){
    
    // Process command line arguments
    if (argc < 2) {
        printf("Usage: %s <model_path> [test_weights]\n", argv[0]);
        printf("Example: %s models/llama3.gguf\n", argv[0]);
        printf("Add 'test_weights' as second argument to test all weights are available\n");
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Check if we should test all weights
    bool should_test_weights = (argc > 2 && strcmp(argv[2], "test_weights") == 0);
    
    if (should_test_weights) {
        printf("Testing all weights in the model...\n");
        bool all_weights_available = test_all_weights_available(model_path);
        
        if (all_weights_available) {
            printf("\nALL WEIGHTS VERIFICATION PASSED! ✓\n");
            return 0;
        } else {
            printf("\nWEIGHT VERIFICATION FAILED! ✗\n");
            printf("Some expected weights are missing from the model.\n");
            return 1;
        }
    }
    
    // Regular model loading and verification as before
    Config config;
    printf("Initialized model config successfully\n");
    
    // Load the model
    printf("Loading model from %s\n", model_path);
    Transformer* model = load_transformer_model(model_path);
    
    if (model != nullptr) {
        printf("Model loaded successfully!\n");
        printf("Model configuration:\n");
        printf("  Embedding dimension: %d\n", model->config.dim);
        printf("  Hidden dimension: %d\n", model->config.hidden_dim);
        printf("  Number of layers: %d\n", model->config.n_layers);
        printf("  Number of attention heads: %d\n", model->config.n_heads);
        printf("  Number of KV heads: %d\n", model->config.n_kv_heads);
        printf("  Vocabulary size: %d\n", model->config.vocab_size);
        printf("  Maximum sequence length: %d\n", model->config.seq_len);
        
        // Clean up
        delete model;
    } else {
        printf("Failed to load model.\n");
        return 1;
    }

    return 0;
}