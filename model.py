import torch
import torch.nn as nn
import math

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



