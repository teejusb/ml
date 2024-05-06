import torch

from .layers import create_transformer


def main():
  # Is CUDA available?
  print(torch.cuda.is_available())


  # Size of the input vocabulary (e.g. English words)
  input_vocab_size = 8500
  # Size of the output vocabulary (e.g. French words)
  output_vocab_size = 8000
  # Size of the input context (e.g. max(len(English sentence)))
  input_context_size = 800
  # Size of the output context. (e.g. max(len(French sentence)))
  output_context_size = 800

  # Define the constants from the paper:
  #
  # Dimension of the model, i.e. the input/output embedding size.
  d_model = 512
  # Number of attention heads. 
  num_heads = 8
  # Number of encoder and decoder layers.
  n = 6
  #  Dimension of the feed-forward network.
  d_ff = 2048
  # Dropout rate.
  dropout = 0.1

  # Create the transformer model.
  transformer = create_transformer(
      input_vocab_size=input_vocab_size,
      output_vocab_size=output_vocab_size,
      input_context_size=input_context_size,
      output_context_size=output_context_size,
      d_model=d_model,
      num_heads=num_heads,
      N=n,
      d_ff=d_ff,
      dropout=dropout
  )


if __name__ == "__main__":
  main() 