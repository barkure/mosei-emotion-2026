# Competition data

Place the original competition attachments in this directory. Only README files are tracked by Git; datasets, labels, and videos remain local. Keep the supplied filenames, sample IDs, labels, and train/validation/test splits unchanged.

```text
data/
  attachment1/
    label-100.xlsx
    <video_id>/<clip_id>.mp4
  attachment2/
    aligned_50.pkl
    unaligned_50.pkl
    label.xlsx
  attachment3/
    aligned/01.pkl ... 30.pkl
    unaligned/01.pkl ... 30.pkl
  attachment4/
    aligned/
      01.pkl ... 20.pkl
      videos/01.mp4 ... 20.mp4
    unaligned/
      01.pkl ... 20.pkl
      videos/01.mp4 ... 20.mp4
```

| Task | Required original data |
| --- | --- |
| Rebuild Q1 features | Attachment 1 labels and videos |
| Train or validate the fixed model | Attachment 2 `aligned_50.pkl` |
| Check both Attachment 2 input interfaces | Both Attachment 2 PKL files |
| Run Q2 specialist inference | Attachment 3 `aligned/` PKL files |
| Run Q3 predictions and feature-position explanations | Attachment 4 `aligned/` PKL files |
| Recover Q3 media evidence | Attachment 4 `aligned/videos/` in addition to its aligned features |

The fixed training, validation, and specialist inference pipeline uses the aligned version. Unaligned files are retained for data inspection, not as replacements for aligned inputs. Fit parameters on Attachment 2 train and select models on valid; Attachments 3 and 4 are for final inference only.

All original attachments together occupy approximately 4.08 GB; Attachment 2 accounts for about 3.89 GB. Existing local copies do not need to be downloaded or copied again.

Run commands from the project root. See the [project README](../README.md#项目运行) and [data interface specification](../src/common/README.md).
