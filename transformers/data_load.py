import time
import os

from collections import Counter

import pandas as pd
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab as make_vocab, Vocab


def build_vocab(path: str, rows: int):
  counter = {'en': Counter(), 'fr': Counter()}
  context_size = {'en': 0, 'fr': 0}
  tokenizer = {
    'en': get_tokenizer('spacy', language='en_core_web_sm'),
    'fr': get_tokenizer('spacy', language='fr_core_news_sm')
  }
  # The training data is too large to fit in memory, so we need to read it in
  # chunks.
  for chunk in pd.read_csv(
      path, usecols=['en', 'fr'], chunksize=1, dtype=str, nrows=rows):
    for lang in ['en', 'fr']:
      line = chunk[lang].values[0]
      tokens = tokenizer[lang](line)
      counter[lang].update(tokens)
      context_size[lang] = max(
        context_size[lang], len(tokens))


  en_vocab = make_vocab(
      counter['en'], specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'])
  fr_vocab = make_vocab(
      counter['fr'], specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'])

  # If a token is not in the vocabulary, return the index of the <UNK> token.
  en_vocab.set_default_index(en_vocab['<UNK>'])
  fr_vocab.set_default_index(fr_vocab['<UNK>'])

  vocab = {
    'en': en_vocab,
    'fr': fr_vocab,
  }
  return vocab, context_size, tokenizer


def encode(s: str, vocab: Vocab, context_size: int, tokenizer: callable,
           add_sos: bool = True, add_eos: bool = True):
  tokens = tokenizer(s)
  if add_sos:
    tokens = ['<SOS>'] + tokens
  if add_eos:
    tokens = tokens + ['<EOS>']

  # Pad the tokens to the context size.
  if len(tokens) < context_size:
    tokens = tokens + ['<PAD>'] * (context_size - len(tokens))

  return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)


def decode(t: torch.Tensor, vocab: Vocab):
  return ' '.join([vocab.itos[token] for token in t])


class ChunkedWmt2014Dataset(Dataset):
  def __init__(self, path: str, vocab: dict[str, Vocab], context_size: int,
               tokenizer: dict[str, callable], chunk_size: int, rows: int):
    self.path = path
    self.vocab = vocab
    self.tokenizer = tokenizer
    self.context_size = context_size
    self.chunk_size = chunk_size
    self.rows = rows

  def __len__(self):
    return self.rows // self.chunk_size
  
  def __getitem__(self, idx):
    chunk = next(
      pd.read_csv(
        self.path,
        names=['en', 'fr'],
        chunksize=self.chunk_size,
        dtype=str,
        skiprows=idx * self.chunk_size + 1,  # Skip the header.
        nrows=self.chunk_size))

    en_lines = chunk['en'].values
    fr_lines = chunk['fr'].values

    # The encoder input.
    en_tensors = []
    # The decoder input.
    fr_tensors = []
    # The decoder output (output labels).
    out_tensors = []

    for en_line, fr_line in zip(en_lines, fr_lines):
      en_tensor = encode(
        en_line,
        self.vocab['en'],
        self.context_size,
        self.tokenizer['en'],
        add_sos=True,
        add_eos=True)
      fr_tensor = encode(
        fr_line,
        self.vocab['fr'],
        self.context_size,
        self.tokenizer['fr'],
        add_sos=True,
        add_eos=False)
      out_tensor = encode(
        fr_line,
        self.vocab['fr'],
        self.context_size,
        self.tokenizer['fr'],
        add_sos=False,
        add_eos=True)

      en_tensors.append(en_tensor)
      fr_tensors.append(fr_tensor)
      out_tensors.append(out_tensor)

    if len(en_tensors) < 64:
      pad_en = torch.full(
        (64 - len(en_tensors), self.context_size),
        self.vocab['en']['<PAD>'],
        dtype=torch.int)

      pad_fr = torch.full(
        (64 - len(fr_tensors), self.context_size),
        self.vocab['fr']['<PAD>'],
        dtype=torch.int)

      pad_out = torch.full(
        (64 - len(out_tensors), self.context_size),
        self.vocab['fr']['<PAD>'],
        dtype=torch.int)
      en_tensors.append(pad_en)
      fr_tensors.append(pad_fr)
      out_tensors.append(pad_out)
    
    return (torch.stack(en_tensors),
            torch.stack(fr_tensors),
            torch.stack(out_tensors))

class Wmt2014Dataset(Dataset):
  def __init__(
      self, path: str, vocab: dict[str, Vocab], context_size: int,
      tokenizer: dict[str, callable]):
    self.path = path
    self.vocab = vocab
    self.tokenizer = tokenizer
    self.context_size = context_size

    self.data = pd.read_csv(path, usecols=['en', 'fr'], dtype=str)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    en_line = row['en']
    fr_line = row['fr']

    en_tensor = encode(
      en_line,
      self.vocab['en'],
      self.context_size,
      self.tokenizer['en'],
      add_sos=True,
      add_eos=True)

    fr_tensor = encode(
      fr_line,
      self.vocab['fr'],
      self.context_size,
      self.tokenizer['fr'],
      add_sos=True,
      add_eos=False)

    out_tensor = encode(
      fr_line,
      self.vocab['fr'],
      self.context_size,
      self.tokenizer['fr'],
      add_sos=False,
      add_eos=True)

    return en_tensor, fr_tensor, out_tensor
    

def load_data(batch_size: int, train_rows: int):
  train_path = os.path.join('..', 'data', 'wmt14_translate_fr-en_train.csv.zip')
  val_path = os.path.join('..', 'data', 'wmt14_translate_fr-en_validation.csv')
  test_path = os.path.join('..', 'data', 'wmt14_translate_fr-en_test.csv')

  # Build the vocab if we haven't already.
  if os.path.exists('vocab.pt') and os.path.exists('context_size.pt'):
    print('Loading vocab/context_size from disk.')
    vocab = torch.load('vocab.pt')
    context_size = torch.load('context_size.pt')
    tokenizer = {
      'en': get_tokenizer('spacy', language='en_core_web_sm'),
      'fr': get_tokenizer('spacy', language='fr_core_news_sm')
    }
  else:
    print('Building vocab.')

    # How long did it take?
    start = time.time()    
    vocab, context_size, tokenizer = build_vocab(train_path, train_rows)
    torch.save(vocab, 'vocab.pt')
    torch.save(context_size, 'context_size.pt')
    print('Vocab built in', time.time() - start, 'seconds.')

  max_context_size = max(context_size.values())
  # Bump up the conetxt size by 2 to account for the <SOS> and <EOS> tokens.
  max_context_size += 2

  train_data = ChunkedWmt2014Dataset(
      train_path, vocab, max_context_size, tokenizer, batch_size, train_rows)
  val_data = Wmt2014Dataset(val_path, vocab, max_context_size, tokenizer)
  test_data = Wmt2014Dataset(test_path, vocab, max_context_size, tokenizer)

  # Don't chunk the training set as it's already chunked.
  train_set = DataLoader(train_data, num_workers=10)
  val_set = DataLoader(val_data, batch_size=batch_size, num_workers=10)
  test_set = DataLoader(test_data, batch_size=batch_size, num_workers=10)

  return train_set, val_set, test_set, vocab, max_context_size