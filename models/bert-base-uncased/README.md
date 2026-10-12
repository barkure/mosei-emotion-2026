# BERT text encoder

Place these files directly in this directory:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
```

Resource: `google-bert/bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594`.

Run `pixi run resources` from the project root to download and verify the pinned files. The complete resource occupies approximately 441 MB and is excluded from Git. It is used for Q1 text features and the frozen text encoder in Q2/Q3.
