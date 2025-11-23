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
    super().__init__()
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
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))
    
  def forward(self, x):
    # x : (batchsize, seq, hiddensize)
    mean = x.mean(dim=-1, keepdim=True) # (batchsize, seq, 1)
    std = x.std(dim=1, keepdim=True) #(batchsize, seq,  1)
    
    return self.alpha * (x-mean)/(std + self.eps) + self.beta
  
    
class FeedForward(nn.Module):
  
  def __init__(self, d_model : int, d_ff : int, dropout : float):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
  def forward(self, x):
    #(batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
    return self.linear_2(self.dropout(self.linear_1(x)))
  
class MultiHeadAttention(nn.Module):

  def __init__(self,d_model : int, num_heads : int ,dropout: float):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_k = d_model // num_heads
    self.w_q = nn.Linear(d_model, d_model, bias=False) # wq
    self.w_k = nn.Linear(d_model, d_model, bias=False) # wk
    self.w_v = nn.Linear(d_model, d_model, bias=False) # wv
    self.w_o = nn.Linear(d_model, d_model, bias=False) #wo
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
    key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
    value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

    #(batch, seq, d_model) --> (batch,seq, num_heads, d_k) --> (batch, num_heads, seq, d_k)
    query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

    #calculate attention
    