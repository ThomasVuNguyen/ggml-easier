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

int main(){
    
    Config config;
    printf("Initialized model config successfully\n");
    TransformerWeights weights;
    printf("Initialized transformer Weights successfully\n");

    return 0;
    
}