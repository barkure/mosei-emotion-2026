# wav2vec2 acoustic model

Place these files directly in this directory:

```text
config.json
model.safetensors
preprocessor_config.json
special_tokens_map.json
tokenizer_config.json
vocab.json
```

Resource: `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.

Run `pixi run resources` from the project root to download and verify the pinned files. The complete resource occupies approximately 378 MB and is excluded from Git. It is used for Q1 word alignment and optional Q3 text-time candidates, not for the core Q2/Q3 prediction step.
