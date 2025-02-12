import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self, 
        d_in, 
        d_out,
        context_length,
        dropout=False, 
        qkv_bias=False
        ):
        '''
        Arguments:
        d_in (int): input dimensions
        d_out (int): output dimensions
        context_length (int)
        dropout (bool): during training, randomly selected hidden layer units are ignored
          to prevent overfitting resulting from excessive reliance on any particular set
          of hidden layer units; this should be set to True only for training
        qkv_bias (bool): whether the layer will learn an additive bias
        
        '''
        
        # initialize weight matrices
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        '''
        register_buffer is used to store information like the causal attention mask
        implemented in this class that should not receive gradients or participate
        in optimization. Essentially, this enables the mask to stay with the model, 
        but not train with it.
        
        Also enables nontrainable data to be efficiently moved across devices (CPU or GPU)
        '''
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length))
            diagonal=1
        )
        
    def forward(self, x):
        '''
        x (tensor): vector embeddings of tokens of input text
        
        Returns a context vector
        '''
        
        b, num_tokns, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        
        attention_scores = queries @ keys.transpose(1, 2) # transpose dimensions 1 and 2, keeping batch dimensions at first position (0)
        
        # In PyTorch, operations with a trailing underscore are performed in place
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
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
        
        if self.dropout:
            attention_weights = self.dropout(attention_weights)
        
        context_vector = attention_weights @ values
        return context_vector