import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    '''
    PyTorch Dataset for the text that will be used to train the model.
    The input text is chunked into sequences and converted into token ids.
    '''
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_token_ids = []
        self.target_token_ids = []
        self.num_tokens = 0
        
        token_ids = tokenizer.encode(text) # tokenize text
        self.num_tokens = len(token_ids)
        
        # chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i+1: i + max_length + 1]
            self.input_token_ids.append(torch.tensor(input_chunk))
            self.target_token_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        # return total number of rows in dataset
        return len(self.input_token_ids)
    
    def __getitem__(self, index):
        # return specified row from dataset
        return self.input_token_ids[index], self.target_token_ids[index]