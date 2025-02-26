import torch
import torch.nn as nn
import attention

class GPTModel(nn.Module):
    def __init__(
        self, 
        vocab_size=50257, 
        context_length=1024, 
        emb_dim=768,
        num_heads=12,
        num_layers=12,
        drop_rate=0.0,
        qkv_bias=False
        ):
        '''
        Arguments:
        vocab_size (int): default is 50,257 words, which is the size of the BPE tokenizer used in the embeddings module
        context_length (int): max numbers of input tokens that can be handled via positional embeddings
        emb_dim (int): embedding size; each token is transformed into 768-dimensional vector by default
        num_heads (int): number of attention heads in multi-head attention mechanism; output of all heads is
          combined in the final model
        num_layers (int): number of transformer blocks in model
        drop_rate (float): during training, randomly selected hidden layer units are ignored
          to prevent overfitting resulting from excessive reliance on any particular set
          of hidden layer units; 10% random drop out of hidden units for training
          This should be set to a positive value only for training
        qkv_bias (bool): whether the layer will learn an additive bias
        '''
        
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout_embedding = nn.Dropout(drop_rate)
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(        
                context_length=context_length, 
                emb_dim=emb_dim,
                num_heads=num_heads,
                drop_rate=drop_rate,
                qkv_bias=qkv_bias) 
             for _ in range(num_layers)])
        
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
        
    def forward(self, in_index):
        batch_size, seq_len = in_index.shape
        token_embeddings = self.token_embedding(in_index)
        # device setting allows training the model on a CPU or GPU
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=in_index.device))
        
        x = token_embeddings + position_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, drop_rate, qkv_bias):
        super().__init__()
        self.attention = attention.Attention(
            d_in = emb_dim,
            d_out = emb_dim,
            context_length = context_length,
            num_heads = num_heads,
            dropout = drop_rate,
            qkv_bias = qkv_bias
        )
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)
        
    def forward(self, x):
        '''
        A shortcut connection creates an alternative shorter path for the gradient to flow through the network by
        skipping one or more layers. This is done by adding the output of one layer to the output of a later layer.
        
        This is done to solve the vanishing gradient problem (i.e., gradients, which guide weight updates during
        training, become progressively smaller as they propagate backwards during the layers). This makes it harder
        to train earlier layers.
        '''
        
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x
    
class LayerNorm(nn.Module):
    '''
    Implements layer normalization to improve stability and efficiency of neural network training.
    
    Vanishing or exploding gradients can make training a neural network difficult. Layer normalization
    adjusts the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1
    (i.e., unit variance). This speeds up convergence to effective weights and ensures consistent,
    reliable training.
    
    Layer normalization is typically applied before and after the multi-head attention module.
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.eps =1e-5 # epsilon--a small constant added to variance to prevent division by 0
        '''
        scale and shift are trainable parameters that the LLM automatically adjusts during 
        training if it would improve the model's performance on the training task
        '''
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    '''
    GELU (Gaussian error linear unit) activation function for TransformerBlock
    
    GELU is smoother than ReLU and also allows small, non-zero outputs for negative
    values. This allows better optimization during training and enables neurons that
    receive negative inputs to still contribute to the learning process.
    '''
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
        
class FeedForward(nn.Module):
    '''
    This is a small neural network, consisting of 2 Linear layers and a GELU activation function,
    that will be used in the TransformerBlock.
    
    The first layer increases the embedding dimension by a factor of 4, enabling improved 
    ability to learn by exploring a richer representation space. This is followed by a GELU
    activation function. The second layer contracts back to the original dimension.
    
    Essentially:
    Linear Layer 1:
        Input (2, 3, 768) -> Output (2, 3, 3072)
    GELU activation
    Linear Layer 2:
        Input (2, 3, 3072) -> Output (2, 3, 768)
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
        