import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
  """Creates a mapping from the input vocabulary to the model's dimension."""
  def __init__(self, vocab_size: int, d_model: int):
    super(InputEmbedding, self).__init__()
    # Size of the vocabulary.
    self.vocab_size = vocab_size
    # Number of dimensions in the model.
    self.d_model = d_model
    # Embedding layer.
    self.embedding = nn.Embedding(vocab_size, d_model)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Take an input tensor and return its corresponding embeddings.

    Args:
      x: A tensor of shape (batch_size, context_size) representing the input.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      embeddings of the input.
    """
    # As mentioned in section 3.4 of the paper, we multiply the embeddings by
    # sqrt(d_model), likely to prevent the embeddings from becoming too small.
    return self.embedding(x) * math.sqrt(self.d_model)
  

class PositionalEncoding(nn.Module):
  """Adds positional encoding to the input.
  
  This is done by using sine and cosine functions of different frequencies tp
  create the positional encoding.

  The even indices use the sine function and the odd indices use the cosine
  function based on the following equation:

  PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) -> even indices
  PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model)) -> odd indices

  where pos is the position within the embedded input, i is ith index,
  and d_model is dimension of the embedding.
  """
  def __init__(self, context_size: int, d_model: int, dropout: float = 0.1):
    super(PositionalEncoding, self).__init__()
    # Max length of the input sequence.
    self.context_size = context_size
    # Number of dimensions in the model.
    self.d_model = d_model
    # Dropout rate.
    self.dropout = nn.Dropout(dropout)

    # Create a positional encoding.
    #                 --> d_model
    #              |[ [0, 0, 0, 0]
    # context_size |  [0, 0, 0, 0]
    #              |  [0, 0, 0, 0]
    #              v  [0, 0, 0, 0] ]
    positional_encoding = torch.zeros(context_size, d_model)

    # Create a tensor that represents positions (0 -> context_size - 1)
    # [0, 1, 2, ..., context_size - 1] -> then becomes:
    #                  --> 1
    #              |[ [0]
    # context_size |  [1]
    #              |  ...
    #              v  [context_size -1] ]
    position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)

    # Dividing by 10000^(2i/d_model) is the same as multiplying by:
    # 1 / 10000^(2i/d_model)
    # 
    # Since 10000^x can get large very quickly, we can rewrite this as:
    #
    # 1 / 10000^(2i/d_model) = (1/10000)^(2i/d_model)
    #                        = e^(log( (1/10000)^(2i / d_model) ))
    #                        = e^( (2i / d_model) * log(1/10000) )
    #                        = e^( 2i * log(1/10000) / (d_model) )
    #                        = e^( 2i * -log(10000) / d_model )
    denominator = torch.exp(
      torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Looks like:
    #                      --> d_model
    #              |[ [s0_0, c0_0, s1_0, c1_0]
    # context_size |  [s0_1, c0_1, s1_1, c1_1]
    #              |  [s0_2, c0_2, s1_2, c1_2]
    #              v  [s0_3, c0_3, s1_3, c1_3] ]
    # for si_pos, ci_pos
    positional_encoding[:, 0::2] = torch.sin(position * denominator)
    positional_encoding[:, 1::2] = torch.cos(position * denominator)

    # Register the positional encoding as a buffer so that it is saved with the
    # model and not treated as a parameter.
    self.register_buffer('positional_encoding',
                         positional_encoding.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Add the positional encoding to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
        
    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      input with the positional encoding added.
    """
    # Add the positional encoding to the input
    x = x + (self.positional_encoding[:, :x.size(1)]).requires_grad_(False)
    return self.dropout(x)


class LayerNormalization(nn.Module):
  """Applies layer normalization to the input.
  
  This is used to normalize the input so that the mean is 0 and the standard
  deviation is 1. This helps the model learn more effectively.
  """
  def __init__(self, epsilon: float = 1e-6):
    super(LayerNormalization, self).__init__()
    # Define a small value to prevent any division by zero.
    self.epsilon = epsilon
    
    # Alpha is a learnable parameter that scales the normalized input.
    self.alpha = nn.Parameter(torch.ones(1))

    # Beta is a learnable parameter that shifts the normalized input.
    self.beta = nn.Parameter(torch.zeros(1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply layer normalization to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
        
      Returns:
        A tensor of shape (batch_size, context_size, d_model) representing the
        normalized input.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    # Normalize the input.
    x = (x - mean) / (std + self.epsilon)
    return self.alpha * x + self.beta
  

class FeedForward(nn.Module):
  """Applies a feed-forward network to the input.

  This feed-forward network consists of two linear transformations with a ReLU
  activation function in between.

  Essentially this is a 3 layer NN that scales the input dimension to a
  larger dimensional space and then scales it back down to the original
  dimension.

  This lets the model learn more complex functions.

  Represented by the following equation:
  FFN(x) = max(0, xW1 + b1)W2 + b2

  input -> linear1 -> ReLU -> linear2 -> output
  """
  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    super(FeedForward, self).__init__()
    # W1 and b1.
    self.linear1 = nn.Linear(d_model, d_ff)
    # Dropout to prevent overfitting.
    self.dropout = nn.Dropout(dropout)
    # W2 and b2.
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the feed-forward network to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the feed-forward network.
    """
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
  """Applies multi-head attention to the input.
  
  This is done by splitting the input into num_heads pieces and applying the
  scaled dot-product attention mechanism to each piece. The outputs are then
  concatenated and a linear transformation is applied to the output.
  """
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
    self.d_keys = d_model // num_heads

    # Define the weights for the linear transformations of the query, key, and
    # value vectors.
    self.w_query = nn.Linear(d_model, d_model)
    self.w_key = nn.Linear(d_model, d_model)
    self.w_value = nn.Linear(d_model, d_model)
    self.w_output = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(
      self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
      mask: torch.Tensor) -> torch.Tensor:
    """Apply scaled dot-product attention to the input.
    
    Args:
      query: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the query.
      key: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the key.
      value: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the value.
      mask: A tensor of shape (batch_size, num_heads, context_size,
        context_size) representing the mask to apply to the attention scores.

    Returns:
      A tensor of shape (batch_size, num_heads, context_size, d_keys)
      representing the output of the scaled dot-product attention.
    """
    # The last dimension of the query, key, and value vectors.
    d_keys = query.size(-1)

    # Calculate Q * K^T / sqrt(d_keys)
    # Only transpose the last two dimensions of key because the first
    # dimension represents the batch size.
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_keys)

    # Apply the mask (if it exists).
    # You usually mask in parallel mode so that we don't let future tokens
    # influence the current token.
    # In sequential mode, we don't need to mask because we don't have future
    # tokens.
    if mask is not None:
      # Before applying the softmax, conditionally apply the mask to
      # hide any values that should not be seen by the model.
      scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply the softmax function to the scores.
    scores = torch.softmax(scores, dim=-1)

    # Apply dropout to the scores.
    scores = self.dropout(scores)

    return scores @ value

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Apply multi-head attention to the input.
    
    Args:
      query: A tensor of shape (batch_size, context_size, d_model) representing
        the query.
      key: A tensor of shape (batch_size, context_size, d_model) representing
        the key.
      value: A tensor of shape (batch_size, context_size, d_model) representing
        the value.
      mask: A tensor of shape (batch_size, context_size) representing the mask
        to apply to the attention scores.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the multi-head attention.
    """    
    # Apply the linear transformations to the query, key, and value vectors.
    query = self.query(query)
    key = self.key(key)
    value = self.value(value)

    # Split the query, key, and value vectors into num_heads pieces.
    query = query.view(
      query.size(0), -1, self.num_heads, self.d_keys).transpose(1, 2)
    key = key.view(
      key.size(0), -1, self.num_heads, self.d_keys).transpose(1, 2)
    value = value.view(
      value.size(0), -1, self.num_heads, self.d_keys).transpose(1, 2)

    output = self.scaled_dot_product_attention(query, key, value, mask)

    # Concatenate the output of the attention heads.
    output = output.transpose(1, 2).contiguous().view(
                output.size(0), -1, self.num_heads * self.d_keys)

    # Apply the final linear transformation.
    return self.w_output(output)
  

class AddAndNorm(nn.Module):
  """Applies the add and norm layer to the input."""
  def __init__(self, dropout: float = 0.1):
    super(AddAndNorm, self).__init__()
    # Dropout to prevent overfitting.
    self.dropout = nn.Dropout(dropout)
    # Layer normalization.
    self.norm = LayerNormalization()


  def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
    """Apply the add and norm layer to the input.
    
    Sublayer is the block preceding this layer that we want to add and normalize
    as well as add the residual connections to.

    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
      sublayer: A nn.Module representing the block preceding this layer.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the add and norm layer.
    """
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

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
              input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
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
              input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
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

  def encode(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    input = self.input_embeddings(input)
    input = self.input_positional_encoding(input)
    return self.encoder(input, mask)
  
  def decode(self, output: torch.Tensor, encoder_output: torch.Tensor,
             input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
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
  input_embeddings = InputEmbedding(input_vocab_size, d_model)
  # Create the input positional encoding.
  input_positional_encoding = PositionalEncoding(input_context_size, d_model)
  # Create the encoder.
  encoder = Encoder(nn.ModuleList([
    EncoderBlock(
      MultiHeadAttention(d_model, num_heads, dropout),
      FeedForward(d_model, d_ff, dropout),
      dropout
    ) for _ in range(N)
  ]))

  # Create the output embeddings.
  output_embeddings = InputEmbedding(output_vocab_size, d_model)
  # Create the output positional encoding.
  output_positional_encoding = PositionalEncoding(output_context_size, d_model)
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
  
  d_model = 512  # Dimension of the model, i.e. the input/output embedding size.
  num_heads = 8  # Number of attention heads.
  n = 6  # Number of encoder and decoder layers.
  d_ff = 2048  # Dimension of the feed-forward network.
  dropout = 0.1  # Dropout rate.


if __name__ == "__main__":
  main() 