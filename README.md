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

The main evaluation script expects a csv-file where the ground truth is standardized (without capital letters or punctuation) in a column called "standardized_text" and predicted text is in a column called "predictions".
It also expects a language code for the written standard ("nob" or "nno").

``` shell
pdm run python -m asr_eval -l nob path/to/your/input_file.csv
```
