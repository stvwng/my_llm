import pytest
import torch
import embeddings
import attention

# to run: pytest tests.py

file_path = "the-verdict.txt"
test_embeddings = embeddings.create_embeddings(file_path=file_path)

def test_embeddings1():
    assert test_embeddings.shape[0] == 4
    
def test_embeddings2():
    assert test_embeddings.shape[1] == 256
    
def test_embeddings3():
    assert test_embeddings.shape[2] == 256
    
def test_attention():
    torch.manual_seed(789)
    sa = Attention() # TODO: add arguments for d_in and d_out
    # TODO: assert output is as expected
    assert 1 == 1
    