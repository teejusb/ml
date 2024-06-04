import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
  """Creates a mapping from the input vocabulary to the model's dimension.
  
  Total Params: vocab_size * d_model
  """
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

  Total Params: 0
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

  Total Params: 2 (alpha + beta)
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

  Total Params = (d_model * d_ff) + d_ff + (d_ff * d_model) + d_model
               = 2 * (d_model * d_ff) + d_ff + d_model
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
    
    assert d_model % num_heads == 0, \
        'Dimension of the model must be divisible by the number of heads.'
    
    # Dimension of the key, query, and value vectors.
    self.d_keys = d_model // num_heads

    # Define the weights for the linear transformations of the query, key, and
    # value vectors.
    self.w_query = nn.Linear(d_model, d_model)
    self.w_key = nn.Linear(d_model, d_model)
    self.w_value = nn.Linear(d_model, d_model)
    self.w_output = nn.Linear(d_model, d_model)

  @staticmethod
  def scaled_dot_product_attention(
      query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
      mask: torch.Tensor, dropout: nn.Dropout) -> torch.Tensor:
    """Apply scaled dot-product attention to the input.
    
    Args:
      query: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the query.
      key: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the key.
      value: A tensor of shape (batch_size, num_heads, context_size, d_keys)
        representing the value.
      mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores.
      dropout: A nn.Dropout layer to apply to the scores.
    Returns:
      A tensor of shape (batch_size, num_heads, context_size, d_keys)
      representing the output of the scaled dot-product attention.
    """
    # The last dimension of the query, key, and value vectors.
    d_keys = query.size(-1)

    # Calculate Q * K^T / sqrt(d_keys)
    #
    # Only transpose the last two dimensions of key because the first
    # two dimensions are the batch size and the number of heads.
    #
    # Shape of scores: (batch_size, num_heads, context_size, context_size)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_keys)

    # Apply the mask (if it exists).
    #
    # You usually mask in parallel mode so that we don't let future tokens
    # influence the current token.
    # In sequential mode, we don't need to mask because we don't have future
    # tokens.
    #
    # The dimension of the mask is the same as the scores.
    if mask is not None:
      # Before applying the softmax, conditionally apply the mask to
      # hide any values that should not be seen by the model.
      scores = scores.masked_fill(
          mask == 0, -1e4 if scores.dtype == torch.float16 else -1e9)
    
    # Apply the softmax function to the scores.
    scores = torch.softmax(scores, dim=-1)

    if dropout is not None:
      # Apply dropout to the scores.
      scores = dropout(scores)

    # Shape of output: (batch_size, num_heads, context_size, d_keys)
    return (scores @ value), scores
  
  def split_heads(self, x: torch.Tensor) -> torch.Tensor:
    """Split the input into num_heads pieces.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.

    Returns:
      A tensor of shape (batch_size, num_heads, context_size, d_keys)
      representing the input split into num_heads pieces.
    """
    batch_size, context_size, _ = x.size()
    return x.view(
        batch_size, context_size, self.num_heads, self.d_keys).transpose(1, 2)
  
  def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
    """Concatenate the input back into a single tensor.
    
    Args:
      x: A tensor of shape
        (batch_size, num_heads, context_size, d_keys) representing
        the input.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      input concatenated back into a single tensor.
    """
    batch_size, _, context_size, _ = x.size()
    # d_model = num_heads * d_keys since d_model is divisible by num_heads.
    return x.transpose(1, 2).contiguous().view(
        batch_size, context_size, self.d_model)

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
      mask: A tensor of shape
        (batch_size, context_size, context_size) representing the
        mask to apply to the attention scores.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the multi-head attention.
    """    
    # Apply the linear transformations to the query, key, and value vectors.
    query = self.split_heads(self.w_query(query))
    key = self.split_heads(self.w_key(key))
    value = self.split_heads(self.w_value(value))

    # Add num_heads to the mask.
    mask = (mask != 0).unsqueeze(1).repeat(1, self.num_heads, 1, 1)

    # Calculate the attention scores.
    output, self.scores = MultiHeadAttention.scaled_dot_product_attention(
        query, key, value, mask, self.dropout)

    # Concatenate the output of the attention heads.
    output = self.concat_heads(output)

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
  """A single encoder block in the transformer.
  
  This block consists of (in order):
  - Multi-head attention layer.
  - Add and norm layer.
  - Feed-forward network.
  - Add and norm layer.
  """
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
    """Apply the encoder block to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
      mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the encoder block.
    """
    # The first block is the multi-head attention layer + add and norm.
    x = self.add_and_norms[0](x, lambda x: self.attention(x, x, x, mask))
    # Then the feed-forward network + add and norm.
    return self.add_and_norms[1](x, self.feed_forward)


class Encoder(nn.Module):
  """A stack of N identical encoder blocks."""
  def __init__(self, layers: nn.ModuleList):
    super(Encoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply each layer of the encoder to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
      mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores.
    
    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the encoder.
    """
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  

class DecoderBlock(nn.Module):
  """A single decoder block in the transformer.

  This block consists of (in order):
  - Masked multi-head attention layer.
  - Add and norm layer.
  - Multi-head attention layer.
  - Add and norm layer.
  - Feed-forward network.
  - Add and norm layer.
  """
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

  def forward(
      self, x: torch.Tensor, encoder_output: torch.Tensor,
      input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
    """Apply the decoder block to the input.

    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
      encoder_output: A tensor of shape (batch_size, context_size, d_model)
        representing the output of the encoder.
      input_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores for the encoder.
      output_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing
        the mask to apply to the attention scores for the decoder.
    
    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the decoder block.
    """
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


class Decoder(nn.Module):
  """A stack of N identical decoder blocks."""
  def __init__(self, layers: nn.ModuleList):
    super(Decoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(
      self, x: torch.Tensor, encoder_output: torch.Tensor,
      input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
    """Apply each layer in the decoder to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
      encoder_output: A tensor of shape (batch_size, context_size, d_model)
        representing the output of the encoder.
      input_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores for the encoder.
      output_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores for the decoder.
        
    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the decoder.
    """
    for layer in self.layers:
      x = layer(x, encoder_output, input_mask, output_mask)
    return self.norm(x)


class PredictionLayer(nn.Module):
  """The final layer of the transformer that predicts the output."""
  def __init__(self, d_model: int, vocab_size: int):
    super(PredictionLayer, self).__init__()
    # Linear transformation.
    self.linear = nn.Linear(d_model, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the prediction layer to the input.
    
    Args:
      x: A tensor of shape (batch_size, context_size, d_model) representing the
        input.
    
    Returns:
      A tensor of shape (batch_size, context_size, vocab_size) representing the
      output of the prediction layer.
    """
    # Apply the linear transformation and the (log) softmax function.
    # The log softmax function is used because it is numerically more stable.
    return torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
  """Tha main transformer model that combines the encoder and decoder."""
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
    """Apply the encoder to the input.
    
    Args:
      input: A tensor of shape (batch_size, context_size) representing the
        input.
      mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the encoder.
    """
    input = self.input_embeddings(input)
    input = self.input_positional_encoding(input)
    return self.encoder(input, mask)
  
  def decode(
      self, output: torch.Tensor, encoder_output: torch.Tensor,
      input_mask: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
    """Apply the decoder to the output.

    Args:
      output: A tensor of shape (batch_size, context_size) representing the
        output.
      encoder_output: A tensor of shape (batch_size, context_size, d_model)
        representing the output of the encoder.
      input_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores for the encoder.
      output_mask: A tensor of shape
        (batch_size, num_heads, context_size, context_size) representing the
        mask to apply to the attention scores for the decoder.

    Returns:
      A tensor of shape (batch_size, context_size, d_model) representing the
      output of the decoder.
    """
    output = self.output_embeddings(output)
    output = self.output_positional_encoding(output)
    return self.decoder(output, encoder_output, input_mask, output_mask)
  
  def predict(self, output: torch.Tensor) -> torch.Tensor:
    """Predict the output.

    Args:
      output: A tensor of shape (batch_size, context_size, d_model) representing
        the output of the decoder.
    
    Returns:
      A tensor of shape (batch_size, context_size, vocab_size) representing the
      output of the prediction layer.
    """
    return self.prediction_layer(output)
  
  def create_mask(self, x: torch.Tensor) -> torch.Tensor:
    """Create a mask to hide padding tokens.
    
    Args:
      x: A tensor of shape (batch_size, context_size) representing the input.
    
    Returns:
      A tensor of shape (batch_size, context_size, context_size)
      representing the mask to apply to the attention scores.
    """
    batch_size = x.size(0)
    size = x.size(1)

    mask = torch.triu(
      torch.ones(batch_size, size, size), diagonal = 1).type(torch.int)

    if torch.cuda.is_available():
      mask = mask.cuda()
    return mask
  
  def forward(
      self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
    """Apply the transformer to the input and output.
    
    Args:
      input: A tensor of shape (batch_size, context_size) representing the
        input.
      output: A tensor of shape (batch_size, context_size) representing the
        output.
    
    Returns:
      A tensor of shape (batch_size, context_size, vocab_size) representing the
      output of the transformer.
    """
    # Create the masks for the encoder and decoder.
    input_mask = self.create_mask(enc_input)
    output_mask = self.create_mask(dec_input)
    # Apply the encoder.
    encoder_output = self.encode(enc_input, input_mask)
    # Apply the decoder.
    decoder_output = self.decode(
        dec_input, encoder_output, input_mask, output_mask)
    # Predict the output.
    return self.predict(decoder_output)


def create_transformer(input_vocab_size: int, output_vocab_size: int,
                       input_context_size: int, output_context_size: int,
                       d_model: int =  512, num_heads: int = 8, N: int = 6,
                       d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
  """Create a transformer model.
  
  Args:
    input_vocab_size: integer, The size of the input vocabulary.
    output_vocab_size: integer, The size of the output vocabulary.
    input_context_size: integer, The size of the input context.
    output_context_size: integer, The size of the output context.
    d_model: integer, The dimension of the model.
    num_heads: integer, The number of attention heads.
    N: integer, The number of encoder and decoder layers.
    d_ff: integer, The dimension of the feed-forward network.
    dropout: Flow, The dropout rate.

  Returns:
    A Transformer model.
  """
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

  if torch.cuda.is_available():
    transformer = transformer.cuda()

  # Initialize the parameters.
  # TODO(teejusb): Understand why this is necessary.
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  
  return transformer