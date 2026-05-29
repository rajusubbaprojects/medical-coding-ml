# scripts/

One-shot generator scripts that write files into `src/` and `prompts/`.

Why this exists: pasting long multi-line content (especially Python with f-strings)
into a terminal via heredoc is unreliable on some shell setups. These scripts use
Python's `Path.write_text()` instead, which is robust.

## Usage pattern

Each `write_<thing>.py` writes one file. Re-run if you change the source content:

```bash
python scripts/write_build_prompt.py
```

## Inventory (Day 5 & 6)

- `write_prompts.py`       -> `prompts/multilabel.txt`, `prompts/primary_dx.txt`
- `write_build_prompt.py`  -> `src/notes/build_prompt.py`
- `write_generate.py`      -> `src/notes/generate.py`
- `write_load_to_bq.py`    -> `src/notes/load_to_bq.py`
- `write_run_day5.py`      -> `src/notes/run_day5.py`
- `write_load_data.py`     -> `src/baseline/load_data.py`

## Ad-hoc tools

- `sample_check.py`        -> prints 3 random notes from the latest run for eyeball QA