import urllib.request
import torch

import dataset
import dataloader

def create_dataloader(
    url = "",
    file_path = "",
    batch_size = 4,
    max_length = 256,
    stride = 128,
    shuffle = True,
    drop_last = True, # drop the last batch if it is shorter than batch_size to prevent loss spikes in training
    num_workers = 0 # number of CPU processes to use for preprocessing
    ):
    '''
    Argument: url as string
    
    Returns dataloader wrapper, allowing access to dataloader
    '''
    
    if url != "":
        temp_file_path = "temp.txt"
        urllib.request.urlretrieve(url, temp_file_path)
    else:
        temp_file_path = file_path
    
    with open(temp_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
    new_dataloader = dataloader.GPTDataLoaderWrapper(
        text=text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return new_dataloader

def create_embeddings(
    url = "",
    file_path = "",
    batch_size = 4,
    max_length = 256,
    stride = 128,
    shuffle = True,
    drop_last = True, # drop the last batch if it is shorter than batch_size to prevent loss spikes in training
    num_workers = 0, # number of CPU processes to use for preprocessing 
    output_dim = 256,
    vocab_size = 50257 # vocab size of BPE tokenizer implmented in dataset module
    ):
    dl_wrapper = create_dataloader(
        url = url,
        file_path = file_path,
        batch_size = batch_size,
        max_length = max_length,
        stride = stride,
        shuffle = shuffle,
        drop_last = drop_last, # drop the last batch if it is shorter than batch_size to prevent loss spikes in training
        num_workers = num_workers, # number of CPU processes to use for preprocessing 
    )
    
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    data_iter = iter(dl_wrapper.dataloader)
    inputs, targets = next(data_iter)
    token_embeddings = token_embedding_layer(inputs)
    
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    
    return token_embeddings + pos_embeddings
    
    
