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

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.residual_one = ResidualConnection(d_model, dropout)
        self.residual_two = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask):
        # Important: in the next call we can clearly see why it's called self-attention.
        # the attention is in fact computed among the input itself.

        x = self.residual_one(x, lambda x: self.mha(x, x, x, dropout=self.dropout, mask=src_mask))
        x = self.residual_two(x, self.ffn)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers

        # why this LayerNorm here? It seems like to not be present in the diagrams of the original paper
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # just the src_mask, since the tgt_mask was already applied in the previous line
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# Project back to vocabulary

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)

        # keep in mind that during training, the whole (batch, seq_len, vocab_size) is useful
        # and the reason is that we are trying to parallelize decoding!
        # during inference instead, we extract only the last token of this (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    # It's a translation task, that's why we need both src_embed and tgt_embed

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


# The following might not be 100% correct because it's a mix of code written by me (donnadamus)
# and the code available at:
# https://github.com/hkproj/pytorch-transformer/blob/main/model.py

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
