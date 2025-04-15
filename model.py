import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Look to the notes on the paper when something is not clear

class InputEmbeddings(nn.Module):

    # d_model is the size of the embedding vector
    # vocab_size is the number of words in the vocabulary

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        # multiply by d_model, as defined in the original paper
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Positional Encoding as introduced in "Attention Is All You Need":
        # For each position pos (0-based) and each dimension i (0-based) of the model:
        #
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # where:
        # - pos is the position (time step) in the sequence.
        # - i is the dimension index.
        # - d_model is the model dimensionality (e.g., 512).
        #
        # The even indices (2i) use sine.
        # The odd indices (2i+1) use cosine.
        #
        # This allows the model to learn relative and absolute positional information

        # The positional encoding matrix should have shape (max_len, d_model)
        # in order to create a positional encoding matrix that can take care
        # of a number of tokens up to max_len

        # Shape (max_len, d_model)

        pe = torch.zeros(max_len, d_model)

        # Shape (max_len, 1)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # slightly modified it for numerical stability

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Shape (1, max_len, d_model)
        pe.unsqueeze(0) # so that I can receive a batch of sentences

        # Store pe (positional encoding tensor) inside the module, but tell PyTorch it’s not a parameter to be learned — just a constant buffer.

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.size(1) to find the real dimension of the sentence
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # Don't learn it
        x = self.dropout(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        # eps is used to avoid 0 at the denominator (look for LayerNorm formula online)
        self.eps = eps

        # Normalizes activations for stable training, but keeps flexibility through learnable scale and shift.
        # Define alpha and bias learnable parameters
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # take average on last dimension (d_model)
        std = x.std(dim=-1, keepdim=True) # compute std on last dimension (d_model)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff) # This already has b1 as well, bias=True
        self.W2 = nn.Linear(d_ff, d_model) # This already has b2 as well, bias=True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W1(x)
        x = nn.functional.relu(x)
        x = self.W2(x)
        x = self.dropout(x)
        return x

# For this particular module look at the paper, in particular the figure
# with each tensor shape

class MultiHeadAttention(nn.Module):
    # h is the number of heads
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        # we want to split the embedding in h heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(self, query, key, value, dropout, mask=None):
        d_k = query.shape[-1]

        # query, key, value have shapes (batch, h, seq_len, d_k)

        # in the next operation, the shapes are:
        # query (batch, h, seq_len, d_k)
        # key (batch, h, d_k, seq_len)

        # For each token in the sequence, we compute its dot product with all key vectors
        # (including itself) using query @ key.transpose(-2, -1), giving us attention scores.
        # This results in a (batch, heads, seq_len, seq_len) tensor where each token has
        # similarity scores with every other token in the sequence.

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # shape (batch, h, seq_len, seq_len)

        # now we apply the mask

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        attention = attention_scores @ value # shape (batch, h, seq_len, d_k)

        return attention, attention_scores

    def forward(self, x, mask=None):
        # This next operation is the same as doing x @ w_q
        # Actually what happens is this:
        # x @ w_q.T
        # but since w_q was stored as (out_features, in_features), so in reverse order
        # doing self.w_q(x) is effectively the same as x @ w_q

        query = self.w_q(x) # (batch, seq_len, d_model)
        key = self.w_k(x) # (batch, seq_len, d_model)
        value = self.w_v(x) # (batch, seq_len, d_model)

        # query is (batch, seq_len, d_model)
        # query.view makes it (batch, seq_len, h, d_k)
        # transpose make the final shape (batch, h, seq_len, d_k)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # -1 lets PyTorch infer seq_len automatically (same as x.shape[1])
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # in the original paper the order of sublayer and norm seems to be swapped,
        # but in many implementations it is as following

        return x + self.dropout(sublayer(self.norm(x)))