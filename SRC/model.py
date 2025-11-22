import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
  def __init__(self, d_model: int,vocab_size: int):
    super().__init__()
    self.d_model = d_model #d_model = Size of each embedding vectors (dimension of embedding)
    self.vocab_size = vocab_size #vocab_size = total number of unique tokens 
    self.embedding = nn.Embedding(vocab_size,d_model) # initializing the embedding for all the tokens of vocabulary

  def forward(self, x):
    return self.embedding(x) *math.sqrt(self.d_model) #normalizing the embedding 

class PositionalEmbedding():