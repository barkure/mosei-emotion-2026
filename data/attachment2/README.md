# Attachment 2: standardized features

Place the supplied files directly in this directory:

| File | Purpose | Approximate size |
| --- | --- | ---: |
| `aligned_50.pkl` | Fixed Q2/Q3 training and validation pipeline | 994 MB |
| `unaligned_50.pkl` | Unaligned data inspection and cross-version interface checks | 2.90 GB |
| `label.xlsx` | Supplied label reference table | Less than 1 MB |

Each PKL preserves the original `train`, `valid`, and `test` dictionaries with 3395, 728, and 727 samples respectively. Read samples as `data[split][field][index]`.

Use train for parameter fitting and valid for model selection. The project does not use test for model selection. Do not substitute the unaligned version in the fixed pipeline. Specialist inference uses the supplied trained checkpoints and does not require this attachment.
