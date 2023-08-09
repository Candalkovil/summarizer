# summarizer
Simple tool to accurately summarize given article


Program works by splitting article into input sentences

Then calculates the embeddings using .encode method

afterwards constructs similarity matrix

Then using BART, it creates a generative summary. Which has been hyper paramter fintuning

decodes output token data
