# BirdCLEF 2026 2nd Place OpenVINO Weights

Upload this folder as a Kaggle Dataset.

Recommended dataset slug:

`birdclef2026-2nd-place-openvino`

The packaged inference notebook defaults to:

`/kaggle/input/birdclef2026-2nd-place-openvino`

If Kaggle gives the dataset a different path, set `WEIGHTS_ROOT` in the first notebook cell to the actual `/kaggle/input/<slug>` path.

Required structure:

```text
outputs/
  class_names.txt
  <model_name>/openvino/<model_name>.xml
  <model_name>/openvino/<model_name>.bin
```
