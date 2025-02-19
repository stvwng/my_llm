import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self, 
        d_in, 
        d_out,
        context_length,
        dropout=0.0,
        num_heads, 
        qkv_bias=False
        ):
        '''
        Arguments:
        d_in (int): input dimensions
        d_out (int): output dimensions
        context_length (int): max numbers of input tokens that can be handled via positional embeddings
        dropout (float): during training, randomly selected hidden layer units are ignored
          to prevent overfitting resulting from excessive reliance on any particular set
          of hidden layer units; 10% random drop out of hidden units for default
          This should be set to a positive value only for training
        num_heads (int): number of "heads", i.e., instances of the self-attention mechanism,
          with its own weights; the output of all heads is combined in the final model.
        qkv_bias (bool): whether the layer will learn an additive bias
        
        '''
        
        # initialize weight matrices
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduce projection dimension to match desired output dimension
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # combine head outputs with a linear layer
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
            torch.triu(torch.ones(context_length, context_length)),
            diagonal=1
        )
        
    def forward(self, x):
        '''
        x (tensor): vector embeddings of tokens of input text
        
        Returns a context vector
        '''
        
        b, num_tokens, d_in = x.shape # tensor shape is (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        
        # split the matrix by adding a num_heads dimension
        # then unroll the last dim: (b, num_tokens, d_out) -> (n, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # transpose from shape (b, num_tokens, num_heads, head_dim) to
        # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # compute dot product for each head
        attention_scores = queries @ keys.transpose(2, 3)
        # truncate mask to number of tokens
        mask_bool = self.mask_bool()[:num_tokens, :num_tokens]
         
        # use mask to fill attention scores
        # In PyTorch, operations with a trailing underscore are performed in place
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
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
        attention_weights = self.dropout(attention_weights)
        
        context_vector = (attention_weights @ values).transpose(1,2) # tensor shape (b, num_tokens, num_heads, head_dim)
        
        # combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vec) # linear projection
        return context_vector