import tiktoken
import torch
from torch.utils.data import DataLoader
import dataset

class GPTDataLoaderWrapper():
    def __init__(
        self,
        text,
        batch_size = 4,
        max_length = 256,
        stride = 128,
        shuffle = True,
        drop_last = True, # drop the last batch if it is shorter than batch_size to prevent loss spikes in training
        num_workers = 0 # number of CPU processes to use for preprocessing
        ):
        
        self.gpt_dataset = None
        self.num_tokens = 0
        self.gpt_dataloader = None

        # instantiate BPE (byte pair encoding) tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        self.gpt_dataset = dataset.GPTDataset(text, tokenizer, max_length, stride)
        self.num_tokens = self.gpt_dataset.num_tokens
        self.gpt_dataloader = DataLoader(
            self.gpt_dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers
        )
        

    
    