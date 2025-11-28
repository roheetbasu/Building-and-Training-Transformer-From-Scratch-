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

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, heads, seq, d_k) @ (batch, heads, d_k, seq)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask shape MUST be broadcastable: (batch, 1, 1, seq)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, heads, seq, seq) @ (batch, heads, seq, d_k)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask):

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq, d_model) → (batch, heads, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key   = key.view(key.shape[0],   key.shape[1],   self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # attention  
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, heads, seq, d_k) → (batch, seq, heads*d_k=d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.num_heads * self.d_k)

        # output projection
        return self.w_o(x)
      
class ResidualConnection(nn.Module):

  def __init__(self, features : int, dropout : float):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization(features)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x))) # (batch, seq, d_model)


class EncoderBlock(nn.Module):

  def __init__(self, features : int, attention_block : MultiHeadAttention, feed_forward_block : FeedForward, dropout : float):
    super().__init__()
    self.attention_block = attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout)for _ in range(2)])

  def forward(self, x, src_mask):
    x = self.residual_connections[0](x, lambda x: self.attention_block(x,x,x,src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x  
  
class Encoder(nn.Module):

  def __init__(self, features : int, layers : nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization(features)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)