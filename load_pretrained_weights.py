import numpy as np
import torch
import torch.nn as nn
import gpt_model
import pretrain_model
import tiktoken
from gpt_download import download_and_load_gpt2

def assign(left, right):
    # helper function to check whether 2 tensors or arrays have the same dimensions or shape
    # returns the right tensor as trainable PyTorch parameters
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        # divide attention and bias weights into 3 equal parts for query, key, and value components
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"],
            3,
            axis=-1
        )
        gpt.transformer_blocks[b].attention.W_query.weight = assign(gpt.transformer_blocks[b].attention.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attention.W_key.weight = assign(gpt.transformer_blocks[b].attention.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attention.W_value.weight = assign(gpt.transformer_blocks[b].attention.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"],
            3,
            axis=-1
        )
        gpt.transformer_blocks[b].attention.W_query.bias = assign(gpt.transformer_blocks[b].attention.W_query.bias, q_b)
        gpt.transformer_blocks[b].attention.W_key.bias = assign(gpt.transformer_blocks[b].attention.W_key.bias, k_b)
        gpt.transformer_blocks[b].attention.W_value.bias = assign(gpt.transformer_blocks[b].attention.W_value.bias, v_b)
        
        gpt.transformer_blocks[b].attention.out_proj.weight = assign(
            gpt.transformer_blocks[b].attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.transformer_blocks[b].attention.out_proj.bias = assign(
            gpt.transformer_blocks[b].attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        
        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
       
    # OpenAI GPT-2 model reused token embedding weights in the output
    # later to reduce the total number of parameters
    # This is known as weight tying. 
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.scale, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
'''
Example:
    
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print("Settings: ", settings)
print()
print("Params keys: ", params.keys())
print("Params token embedding weights: ", params["wte"])
print("Token embedding weights shape: ", params["wte"].shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

gpt = gpt_model.GPTModel(
    vocab_size = 50257,
    context_length = 1024,
    emb_dim = 768,
    num_heads = 12,
    num_layers = 12,
    drop_rate = 0.0,
    qkv_bias = True
)
gpt.eval()
load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = pretrain_model.generate(
    model=gpt,
    index=pretrain_model.text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=1024,
    top_k=50,
    temperature=1.5
)

print("Output:\n", pretrain_model.token_ids_to_text(token_ids, tokenizer))
'''
