# ASR Eval

Code to do yearly evaluation of Norwegian speech recognition models

## Install

Use uv or pdm to install dependencies from `pyproject.toml`

```shell
pdm install
```

## Predict

``` shell
pdm run python -m asr_eval.predict <args>
```

## Evaluate speech recognition predictions

The main evaluation script expects a csv-file where the ground truth is standardized (without capital letters or punctuation) in a column called "standardized_text" (with the langauge code flag  `-l nob`) or "standardized_text_nn" (with the langauge code flag `-l nno`) and predicted text is in a column called "predictions".

``` shell
pdm run python -m -l nno asr_eval path/to/your/input_file_with_nynorsk_predictions.csv
```
