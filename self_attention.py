import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        '''
        Arguments:
        d_in (int): input dimensions
        d_out (int): output dimensions
        qkv_bias (bool): whether the layer will learn an additive bias
        
        Returns a context vector
        '''
        
        # initialize weight matrices
        super().__init()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        '''
        x (tensor): vector embeddings of tokens of input text
        '''
        
        keys = self.W_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        
        attention_scores = queries @ keys.T
        
        '''
        Scaling the attention scores by the square root of the embedding dimension of the keys 
        (i.e., raising to the 0.5 power) improves training performance by avoiding small 
        gradients. Large dot products can result in very small gradients during
        backpropagation when softmax is applied. As dot products increase, softmax can 
        behave nore like a step function, resulting in near-zero gradients. These, in turn,
        can slow down learning or cause learning to stagnate.
        '''
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector