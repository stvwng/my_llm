import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(
        self, 
        vocab_size=50257, 
        context_length=1024, 
        emb_dim=768,
        num_heads=12,
        num_layers=12,
        drop_rate=0.1,
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
          of hidden layer units; 10% random drop out of hidden units for default
          This should be set to a positive value only for training
        qkv_bias (bool): whether the layer will learn an additive bias
        '''
        
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout_embedding = nn.Dropout(drop_rate)
        self.transformer_blocks = nn.Sequential(
            [TransformerBlock(        
                vocab_size=50257, 
                context_length=1024, 
                emb_dim=768,
                num_heads=12,
                num_layers=12,
                drop_rate=0.1,
                qkv_bias=False) 
             for _ in range(num_layers)])
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
        
    def forward(self, in_index):
        batch_size, seq_len = in_index.shape
        token_embeddings = self.token_embedding(in_index)
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=in_index.device))
        
        x = token_embeddings + position_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_form(x)
        logits = self.out_head(x)
        return logits
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
    def forward(self, x):
        return x
        