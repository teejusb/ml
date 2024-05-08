import torch
import os

import pandas as pd

from torch.utils.data import DataLoader

from data_load import load_data, decode
from layers import create_transformer


def main():
  # Is CUDA available?
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True

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

  batch_size = 64

  # How many training examples to use.
  train_rows = 32768

  train, val, test, vocab, context_size = load_data(
      batch_size=batch_size, train_rows=train_rows)
  
  # Size of the input vocabulary (e.g. English words)
  input_vocab_size = len(vocab['en'])
  # Size of the output vocabulary (e.g. French words)
  output_vocab_size = len(vocab['fr'])
  # Size of the input context (e.g. max(len(English sentence)))
  input_context_size = context_size
  # Size of the output context. (e.g. max(len(French sentence)))
  output_context_size = context_size

  # Print Constants
  print(f"Input vocab size: {input_vocab_size}")
  print(f"Output vocab size: {output_vocab_size}")
  print(f"Input context size: {input_context_size}")
  print(f"Output context size: {output_context_size}")

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

  checkpoint_interval = 10
  epochs = 100
  
  # Load from checkpoint if we haven't run through all the epochs
  checkpoint = None
  for epoch in range(epochs, 0, -checkpoint_interval):
    try:
      transformer.load_state_dict(torch.load(f"transformer_epoch_{epoch}.pth"))
      checkpoint = epoch
      print(f"Loaded checkpoint from epoch {epoch}")
      break
    except FileNotFoundError:
      continue

  loss_fn = torch.nn.CrossEntropyLoss(
    ignore_index=vocab['fr']['<PAD>']).to(device)
  optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, eps=1e-9)

  if not checkpoint or checkpoint != epochs:
    # Model has not been trained yet.
    if not checkpoint:
      checkpoint = 1

    for epoch in range(checkpoint, epochs+1):
      # Training
      train_loss = 0.0
      transformer.train()
      for enc_input, dec_input, label in train:
        # Training data is pre-chunked, so input_data comes in as
        # (1, batch_size, context_size)
        # We need to remove the first dimension.
        enc_input = torch.squeeze(enc_input, dim=0)
        dec_input = torch.squeeze(dec_input, dim=0)
        label = torch.squeeze(label, dim=0)

        # Transfer to GPU.
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = transformer(enc_input, dec_input)
        loss = loss_fn(output.contiguous().view(
            -1, output_vocab_size), label.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

      # Validation
      val_loss = 0.0
      transformer.eval()
      for enc_input, dec_input, label in val:
        enc_input = torch.squeeze(enc_input, dim=0)
        dec_input = torch.squeeze(dec_input, dim=0)
        label = torch.squeeze(label, dim=0)

        # Transfer to GPU.
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        label = label.to(device)

        output = transformer(enc_input, dec_input)
        loss = loss_fn(output.contiguous().view(
            -1, output_vocab_size), label.contiguous().view(-1))
        val_loss += loss.item()

      if epoch % checkpoint_interval == 0:
        torch.save(transformer.state_dict(), f"transformer_epoch_{epoch}.pth")
      print(f"Epoch {epoch}\t\t"
            "Training loss: {train_loss / len(train)}\t\t"
            "Validation loss: {val_loss / len(val)}")

  # Make some predictions
  transformer.eval()
  with torch.no_grad():
    for enc_input, dec_input, label in test:
      enc_input = torch.squeeze(enc_input, dim=0)
      dec_input = torch.squeeze(dec_input, dim=0)
      label = torch.squeeze(label, dim=0)

      output = transformer(enc_input, dec_input)
      print("Input:", decode(enc_input[0], vocab['en']))
      print("Prediction:", decode(output[0].argmax(dim=-1), vocab['fr']))
      print("Actual:", decode(label[0], vocab['fr']))


if __name__ == "__main__":
  main() 