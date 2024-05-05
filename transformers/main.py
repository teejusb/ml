import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super(InputEmbedding, self).__init__()
    # Number of dimensions in the model.
    self.d_model = d_model
    # Size of the vocabulary.
    self.vocab_size = vocab_size
    # Embedding layer.
    self.embedding = nn.Embedding(vocab_size, d_model)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # To prevent the embeddings from becoming too small, multiply the embeddings
    # by sqrt(d_model).
    return self.embedding(x) * math.sqrt(self.d_model)
  

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, context_size: int, dropout: float = 0.1):
    super(PositionalEncoding, self).__init__()
    # Number of dimensions in the model.
    self.d_model = d_model
    # Max length of the input sequence.
    self.context_size = context_size
    # Dropout rate.
    self.dropout = nn.Dropout(dropout)

    # Create a positional encoding.
    #                   --> d_model
    #              |
    # context_size |
    #              v
    positional_encoding = torch.zeros(context_size, d_model)

    # Create a tensor that represents positions (0 -> context_size - 1)
    # [0, 1, 2, ..., context_size - 1] -> then becomes:
    # [[0], [1], [2], ..., [context_size - 1]]
    position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)

    # The paper uses sine and cosine functions of different frequencies
    # to create the positional encoding.
    #
    # The even indices use the sine function and the odd indices use the cosine
    # function.
    # sin(pos / 10000^(2i/d_model)) -> even indices
    # cos(pos / 10000^(2i/d_model)) -> odd indices
    # 
    # -log(10000) / d_model = (log(1/10000) / d_model)
    # log(1/10000) / d_model = log( (1/100000) ^ (1/d_model) )
    # 2i * log( (1/100000) ^ (1/d_model) ) = log( (1/10000)^(2i / d_model) )
    # e^( log( (1/10000)^(2i / d_model) ) = (1/10000)^(2i/d_model)
    # (1/10000)^(2i/d_model) = 1 / 10000^(2i/d_model)
    denominator = torch.exp(
      torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    positional_encoding[:, 0::2] = torch.sin(position * denominator)
    positional_encoding[:, 1::2] = torch.cos(position * denominator)

    # Add a batch dimension to the positional encoding.
    # unsqueeze converts it back to:
    # [sin_0, cos_0, sin_1, cos_1, ..., sin_context_size, cos_context_size]
    positional_encoding = positional_encoding.unsqueeze(0)

    # Register the positional encoding as a buffer.
    self.register_buffer('positional_encoding', positional_encoding)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Add the positional encoding to the input
    x = x + (self.positional_encoding[:, :x.size(1), :]).requires_grad_(False)
    return self.dropout(x)


class LayerNormalization(nn.Module):
  def __init__(self, epsilon: float = 1e-6):
    super(LayerNormalization, self).__init__()
    # Define a small value to prevent division by zero.
    self.epsilon = epsilon
    
    # Alpha is a learnable parameter that scales the normalized input.
    self.alpha = nn.Parameter(torch.ones(1))

    # Beta is a learnable parameter that shifts the normalized input.
    self.beta = nn.Parameter(torch.zeros(1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    # Normalize the input.
    if std == 0:
      std = std + self.epsilon
    x = (x - mean) / (std)
    return self.alpha * x + self.beta
  

# Feed-Forward Network:
# This feed-forward network consists of two linear transformations
# with a ReLU activation function in between.
#
# Essentially is a 3 layer NN that converts the input dimension to
# a smaller dimension and then back to the original dimension.
#
# This lets the model learn more complex functions.
#
# Represented by the following equation:
# FFN(x) = max(0, xW1 + b1)W2 + b2
#
# input -> linear1 -> ReLU -> linear2 -> output
class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    super(FeedForward, self).__init__()
    # W1 and b1.
    self.linear1 = nn.Linear(d_model, d_ff)
    # Dropout to prevent overfitting.
    self.dropout = nn.Dropout(dropout)
    # W2 and b2.
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
    super(MultiHeadAttention, self).__init__()
    # Number of dimensions in the model.
    self.d_model = d_model
    # Number of attention heads.
    self.num_heads = num_heads
    # Dropout to prevent overfitting.
    self.dropout = nn.Dropout(dropout)
    
    assert (d_model % num_heads == 0,
            'Dimension of the model must be divisible by the number of heads.')
    
    # Dimension of the key, query, and value vectors.
    self.d_k = d_model // num_heads

    # Define the weights for the linear transformations of the query, key, and
    # value vectors.
    self.w_query = nn.Linear(d_model, d_model)
    self.w_key = nn.Linear(d_model, d_model)
    self.w_value = nn.Linear(d_model, d_model)
    self.w_output = nn.Linear(d_model, d_model)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask: bool) -> torch.Tensor:
    def scaled_dot_product_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: bool) -> torch.Tensor:
      # The last dimension of the query, key, and value vectors.
      d_k = query.size(-1)

      # Calculate Q * K^T / sqrt(d_k)
      # Only transpose the last two dimensions of key because the first
      # dimension represents the batch size.
      scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

      # Apply the mask (if it exists).
      if mask is not None:
        # Before applying the softmax, conditionally apply the mask to
        # hide any values that should not be seen by the model.
        scores = scores.masked_fill(mask == 0, -1e9)
      
      # Apply the softmax function to the scores.
      scores = torch.softmax(scores, dim=-1)

      # Apply dropout to the scores.
      scores = self.dropout(scores)

      return scores @ value
    
    # Apply the linear transformations to the query, key, and value vectors.
    query = self.query(query)
    key = self.key(key)
    value = self.value(value)

    # Split the query, key, and value vectors into num_heads pieces.
    query = query.view(
      query.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
    key = key.view(
      key.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
    value = value.view(
      value.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    output = scaled_dot_product_attention(query, key, value, mask)

    # Concatenate the output of the attention heads.
    output = output.transpose(1, 2).contiguous().view(output.size(0), -1,
                                                      self.num_heads * self.d_k)

    # Apply the final linear transformation.
    return self.w_output(output)
  

class AddAndNorm(nn.Module):
  def __init__(self, dropout: float = 0.1):
    super(AddAndNorm, self).__init__()
    # Dropout to prevent overfitting.
    self.dropout = nn.Dropout(dropout)
    # Layer normalization.
    self.norm = LayerNormalization()

  # Sublayer is the block preceding this layer that we want to add and normalize
  # as well as add the residual connections to.
  def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
    return x + self.dropout(sublayer(self.norm(x)))
  

class EncoderBlock(nn.Module):
  def __init__(self, attention: MultiHeadAttention, feed_forward: FeedForward,
               dropout: float = 0.1):
    super(EncoderBlock, self).__init__()
    # Multi-head attention.
    self.attention = attention
    # Feed-forward network.
    self.feed_forward = feed_forward
    # Add and norm layers.
    self.add_and_norms = nn.ModuleList([
      AddAndNorm(dropout) for _ in range(2)
    ])

  def forward(self, x: torch.Tensor, mask: bool) -> torch.Tensor:
    # The first block is the multi-head attention layer + add and norm.
    x = self.add_and_norms[0](x, lambda x: self.attention(x, x, x, mask))
    # Then the feed-forward network + add and norm.
    return self.add_and_norms[1](x, self.feed_forward)


# The encoder is a stack of N identical encoder blocks.
class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super(Encoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x: torch.Tensor, mask: bool) -> torch.Tensor:
    # Apply each layer in the encoder to the input.
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  

class DecoderBlock(nn.Module):
  def __init__(self,
               masked_attention: MultiHeadAttention,
               attention: MultiHeadAttention,
               feed_forward: FeedForward,
               dropout: float = 0.1):
    super(DecoderBlock, self).__init__()
    # Masked multi-head attention.
    self.masked_attention = masked_attention
    # Multi-head attention.
    self.attention = attention
    # Feed-forward network.
    self.feed_forward = feed_forward
    # Add and norm layers.
    self.add_and_norms = nn.ModuleList([
      AddAndNorm(dropout) for _ in range(3)
    ])

  def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
              input_mask: bool, output_mask: bool) -> torch.Tensor:
    # The first block is the masked multi-head attention layer + add and norm.
    x = self.add_and_norms[0](x, lambda x: self.masked_attention(
        x, x, x, output_mask))
    # The second block is the multi-head attention layer + add and norm.
    # The query comes from the previous block. The key and value comes from the
    # encoder.
    x = self.add_and_norms[1](x, lambda x: self.attention(
        x, encoder_output, encoder_output, input_mask))
    # Then the feed-forward network + add and norm.
    return self.add_and_norms[2](x, self.feed_forward)


# The decoder is a stack of N identical decoder blocks.
class Decoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super(Decoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
              input_mask: bool, output_mask: bool) -> torch.Tensor:
    # Apply each layer in the decoder to the input.
    for layer in self.layers:
      x = layer(x, encoder_output, input_mask, output_mask)
    return self.norm(x)


class PredictionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super(PredictionLayer, self).__init__()
    # Linear transformation.
    self.linear = nn.Linear(d_model, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Apply the linear transformation and the (log) softmax function.
    # The log softmax function is used because it is numerically more stable.
    return torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.module):
  def __init__(self,
               input_embeddings: InputEmbedding,
               input_positional_encoding: PositionalEncoding,
               encoder: Encoder,
               output_embeddings: InputEmbedding,
               output_positional_encoding: PositionalEncoding,
               decoder: Decoder,
               prediction_layer: PredictionLayer):
    super(Transformer, self).__init__()
    # All encoder layers.
    self.input_embeddings = input_embeddings
    self.input_positional_encoding = input_positional_encoding
    self.encoder = encoder

    # All decoder layers.
    self.output_embeddings = output_embeddings
    self.output_positional_encoding = output_positional_encoding
    self.decoder = decoder
    self.prediction_layer = prediction_layer

  def encode(self, input: torch.Tensor, mask: bool) -> torch.Tensor:
    input = self.input_embeddings(input)
    input = self.input_positional_encoding(input)
    return self.encoder(input, mask)
  
  def decode(self, output: torch.Tensor, encoder_output: torch.Tensor,
             input_mask: bool, output_mask: bool) -> torch.Tensor:
    output = self.output_embeddings(output)
    output = self.output_positional_encoding(output)
    return self.decoder(output, encoder_output, input_mask, output_mask)
  
  def predict(self, output: torch.Tensor) -> torch.Tensor:
    return self.prediction_layer(output)



def create_transformer(input_vocab_size: int, output_vocab_size: int,
                       input_context_size: int, output_context_size: int,
                       d_model: int, num_heads: int, N: int, d_ff: int,
                       dropout: float) -> Transformer:
  # Create the input embeddings.
  input_embeddings = InputEmbedding(d_model, input_vocab_size)
  # Create the input positional encoding.
  input_positional_encoding = PositionalEncoding(d_model, input_context_size)
  # Create the encoder.
  encoder = Encoder(nn.ModuleList([
    EncoderBlock(
      MultiHeadAttention(d_model, num_heads, dropout),
      FeedForward(d_model, d_ff, dropout),
      dropout
    ) for _ in range(N)
  ]))

  # Create the output embeddings.
  output_embeddings = InputEmbedding(d_model, output_vocab_size)
  # Create the output positional encoding.
  output_positional_encoding = PositionalEncoding(d_model, output_context_size)
  # Create the decoder.
  decoder = Decoder(nn.ModuleList([
    DecoderBlock(
      MultiHeadAttention(d_model, num_heads, dropout),
      MultiHeadAttention(d_model, num_heads, dropout),
      FeedForward(d_model, d_ff, dropout),
      dropout
    ) for _ in range(N)
  ]))

  # Create the prediction layer.
  prediction_layer = PredictionLayer(d_model, output_vocab_size)

  transformer = Transformer(
    input_embeddings, input_positional_encoding, encoder,
    output_embeddings, output_positional_encoding, decoder,
    prediction_layer
  )

  # Initialize the parameters.
  # TODO(teejusb): Understand why this is necessary.
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  
  return transformer


def main():
  # print cuda is available
  print(torch.cuda.is_available())

  # Define the constants from the paper:
  d_model = 512
  num_heads = 8
  n = 6
  d_ff = 2048
  dropout = 0.1


if __name__ == "__main__":
  main() 