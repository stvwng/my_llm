import gpt_model
import torch

torch.manual_seed(123)
x = torch.rand(2, 4, 768)

vocab_size = 50257
context_length = 1024
emb_dim = 768
num_heads = 12
num_layers = 12
drop_rate = 0.1
qkv_bias = False

block = gpt_model.TransformerBlock(emb_dim=emb_dim, context_length=context_length, num_heads=num_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
output = block(x)

print("Input shape: ", x.shape)
print("Output shape: ", output.shape)