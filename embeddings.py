import urllib.request
import torch

import dataset
import dataloader

def create_dataloader_wrapper(
    url = "",
    file_path = "",
    batch_size = 4,
    max_length = 256,
    stride = 128,
    shuffle = True,
    drop_last = True,
    num_workers = 0 
    ):
    '''
    Arguments: 
    url (string): url of text to use to train model;
    file_path (string): file path of text to use to train model;
    Either url or file_path must be specified
    
    batch_size (int): number of samples in each batch
    max_length (int): max number of tokens in a sample
    stride (int): how much to "slide" the input window when creating a new batch
    shuffle (boolean): shuffle data at every epoch
    drop_last (boolean): drop the last batch if it is shorter than batch_size to prevent loss spikes in training
    num_workers: number of CPU processes to use for preprocessing
    
    
    Returns GPTDataLoaderWrapper, allowing access to gpt_dataloader (PyTorch DataLoader instance)
    '''
    
    if url != "":
        temp_file_path = "temp.txt"
        urllib.request.urlretrieve(url, temp_file_path)
    elif file_path != "":
        temp_file_path = file_path
    else:
        raise ValueError("Specify a url or file_path for text to use to train model")
    
    with open(temp_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
    new_dataloader_wrapper = dataloader.GPTDataLoaderWrapper(
        text=text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return new_dataloader_wrapper

def create_embeddings(
    url = "",
    file_path = "",
    batch_size = 4,
    max_length = 256,
    stride = 128,
    shuffle = True,
    drop_last = True,
    num_workers = 0, 
    output_dim = 256,
    vocab_size = 50257
    ):
    '''
    Arguments: 
    url (string): url of text to use to train model;
    file_path (string): file path of text to use to train model;
    Either url or file_path must be specified
    
    batch_size (int): number of samples in each batch
    max_length (int): max number of tokens in a sample
    stride (int): how much to "slide" the input window when creating a new batch
    shuffle (boolean): shuffle data at every epoch
    drop_last (boolean): drop the last batch if it is shorter than batch_size to prevent loss spikes in training
    num_workers (int): number of CPU processes to use for preprocessing
    output_dim (int): for each token, the number of dimensions that it is embedded into;
    e.g., the default is 256, so each token is embedded into a 256-dimensional vector
    vocab_size (int): vocab size of tokenizer; the 
    default BPE tokenizer in the dataset module has a vocab size of 50257 tokens
    
    
    Returns a PyTorch tensor of the embeddings for the input text to be used for training the model.
    '''
    dl_wrapper = create_dataloader_wrapper(
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
    data_iter = iter(dl_wrapper.gpt_dataloader)
    inputs, targets = next(data_iter)
    token_embeddings = token_embedding_layer(inputs)
    
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    
    return token_embeddings + pos_embeddings
    
    
