# Implemenation of the Transformer Architecture

Following the "Attention is All You Need" paper, this project re-implements the
architecture and data loading pipelines to *supposedly* train an en->fr
translator.

I did reference this [Kaggle Notebook](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch)
when building out my version as it was very helpful in walking through the
original paper.

While the paper used the WMT 2014 English-to-German dataset, I opted for the
WMT 2014 English-to-French dataset as I was personally more familiar with it.

The data set can be found on Hugging Face [here](https://huggingface.co/datasets/wmt/wmt14).
For use with the repo, you can download the dataset, and place it in the parent
`data` directory.

## Conclusion

Unfortunately since my machine only has 16GB of Memory and 12 GB of VRAM
(an RTX 4070), I don't have enough compute to actually train this model.

To avoid Out-of-Memory issues, I set the default `dtype` to `float16` using:
```py
torch.set_default_dtype(torch.float16)
```

I also reduced `d_model` from `512`->`256` and `d_ff` from `2048`->`1024`.
I also only loaded `32,768` examples for training.

I was able to report losses (no `nans`), but it did not converge after 100
epochs.

I suspect with more compute and a wider `dtype` this would work.
