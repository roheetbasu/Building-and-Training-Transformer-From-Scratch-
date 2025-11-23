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

class PositionalEncoding(nn.Module):
  def __init__(self,d_model :int,seq : int, dropout : float):
    self.d_model = d_model
    self.seq = seq
    self.dropout = nn.Dropout(dropout)

    # create a matrix of shape (seq, d_model)
    pe = torch.zeros(seq,d_model)

    # create a vector of shape (seq)
    position = torch.arange(0, seq, d_type=torch.float).unsqueeze(1)

    # create a vector of shape (d_model)
    div_term = torch.exp(torch.arange(0, d_model,2).float * -(math.log(10000.0) / d_model))

    #apply sine to indices
    pe[:, 0::2] = torch.sin(position * div_term)

    #apply cosine to indices
    pe[:,1::2] = torch.cos(position * div_term)

    #add a batch dimension to the positional encoding 
    pe = pe.unsqueeze(0)

    #register positional encoding as buffer
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False) # (batch, seq, d_model)
    return self.dropout(x)
  
class LayerNormalization(nn.Module):
  
  def __init__(self,features: int, eps : float = 10 ** -6):
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))
    
  def forward(self, x):
    # x : (batchsize, seq, hiddensize)
    mean = x.mean(dim=-1, keepdim=True) # (batchsize, seq, 1)
    std = x.std(dim=1, keepdim=True) #(batchsize, seq,  1)
    
    return self.alpha * (x-mean)/(std + self.eps) + self.beta
  
    
    