import pytest
import embeddings

file_path = "the-verdict.txt"
test_embeddings = embeddings.create_embeddings(file_path=file_path)

def test_embeddings1():
    assert test_embeddings.shape[0] == 4
    
def test_embeddings2():
    assert test_embeddings.shape[1] == 256
    
def test_embeddings3():
    assert test_embeddings.shape[2] == 256