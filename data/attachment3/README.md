# Attachment 3: missing-modality specialist inputs

Keep the two supplied versions in separate directories:

- [aligned/](aligned/README.md): `01.pkl` through `30.pkl`, used by the fixed Q2 inference pipeline.
- [unaligned/](unaligned/README.md): `01.pkl` through `30.pkl`, retained for input inspection.

These samples have no labels and are used only for final inference. Preserve zero-padded filenames: the file number identifies each prediction. Do not fill missing spans manually or use these inputs to tune the model.
