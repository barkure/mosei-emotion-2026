# Attachment 1: raw videos and labels

Place `label-100.xlsx` and the original video directories here:

```text
attachment1/
  label-100.xlsx
  <video_id>/
    <clip_id>.mp4
```

For example, `-egA8-b7-3M/9.mp4` must keep its original directory and filename. The supplied data contains 100 clips across 37 video directories. Do not replace, remove, or relabel abnormal samples.

These files are used to rebuild Q1 features; raw media processing also requires FFmpeg. The [sample inventory](../../outputs/q1_features/attachment1_inventory.csv) records the expected clip paths. Generated features belong in `outputs/q1_features/`, not here.
