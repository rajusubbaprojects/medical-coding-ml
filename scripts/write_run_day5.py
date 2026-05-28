from pathlib import Path

CONTENT = '''"""Day 5 driver: generate synthetic discharge notes end-to-end.

Pipeline:
    1. fetch_patients          -> list[patient_dict]
    2. build_*_prompt          -> str
    3. generate_note           -> {"text", "input_tokens", ...}
    4. assemble BQ rows + JSONL backup
    5. load to BigQuery
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.notes.build_prompt import build_multilabel_prompt, build_primary_dx_prompt
from src.notes.generate import generate_note, MODEL
from src.notes.load_to_bq import ensure_tables, insert_notes
from src.notes.select_patients import fetch_patients

PROMPT_VERSION_MULTI = "v1-multilabel"
PROMPT_VERSION_PRIMARY = "v1-primary_dx"

BACKUP_DIR = Path("data/notes_synth")

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_note_id(run_id: str, patient_id: str, strategy: str) -> str:
    return f"{run_id}_{patient_id}_{strategy}"


def build_rows_for_note(
    *,
    run_id: str,
    patient: dict[str, Any],
    strategy: str,
    prompt_version: str,
    generation: dict[str, Any],
    primary_code: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one notes_synth row + N notes_labels rows for a single note."""
    note_id = make_note_id(run_id, patient["patient_id"], strategy)
    labels = [c["icd10_code"] for c in patient["conditions"]]

    note_row = {
        "note_id": note_id,
        "run_id": run_id,
        "patient_id": patient["patient_id"],
        "strategy": strategy,
        "prompt_version": prompt_version,
        "model": generation["model"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note_text": generation["text"],
        "labels": labels,
        "primary_code": primary_code,
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "finish_reason": generation["finish_reason"],
    }

    label_rows = [
        {
            "note_id": note_id,
            "icd10_code": c["icd10_code"],
            "icd10_description": c["icd10_description"],
            "is_primary": (c["icd10_code"] == primary_code),
        }
        for c in patient["conditions"]
    ]

    return note_row, label_rows


def run(*, n_patients: int, strategies: list[str], dry_run: bool) -> None:
    run_id = make_run_id()
    logger.info("=== Run %s ===", run_id)
    logger.info("Patients: %d  Strategies: %s  Dry-run: %s",
                n_patients, strategies, dry_run)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"run_{run_id}.jsonl"

    if not dry_run:
        ensure_tables()

    patients = fetch_patients(n_patients=n_patients)
    logger.info("Fetched %d patients.", len(patients))

    all_note_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []  # (patient_id, strategy, error)

    total_calls = len(patients) * len(strategies)
    pbar = tqdm(total=total_calls, desc="generating")

    with open(backup_path, "a") as backup:
        for p in patients:
            for strategy in strategies:
                try:
                    if strategy == "multilabel":
                        prompt = build_multilabel_prompt(p)
                        primary_code = None
                        prompt_version = PROMPT_VERSION_MULTI
                    elif strategy == "primary_dx":
                        prompt, primary = build_primary_dx_prompt(p)
                        primary_code = primary["icd10_code"]
                        prompt_version = PROMPT_VERSION_PRIMARY
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    gen = generate_note(prompt)

                    note_row, label_rows = build_rows_for_note(
                        run_id=run_id,
                        patient=p,
                        strategy=strategy,
                        prompt_version=prompt_version,
                        generation=gen,
                        primary_code=primary_code,
                    )

                    all_note_rows.append(note_row)
                    all_label_rows.extend(label_rows)

                    backup.write(json.dumps(note_row) + "\\n")
                    backup.flush()

                except Exception as e:
                    logger.exception("Failed: patient=%s strategy=%s",
                                     p["patient_id"], strategy)
                    failures.append((p["patient_id"], strategy, str(e)))
                finally:
                    pbar.update(1)

    pbar.close()

    logger.info("Generated: %d notes (%d label rows). Failures: %d",
                len(all_note_rows), len(all_label_rows), len(failures))
    logger.info("Backup: %s", backup_path)

    if failures:
        for pid, strat, err in failures[:5]:
            logger.warning("  failure: %s / %s -> %s", pid, strat, err[:100])

    total_in = sum(r["input_tokens"] or 0 for r in all_note_rows)
    total_out = sum(r["output_tokens"] or 0 for r in all_note_rows)
    logger.info("Tokens: %d in + %d out", total_in, total_out)
    # Rough cost @ Gemini 2.5 Flash list prices ($0.30/M in, $2.50/M out):
    cost = total_in * 0.30 / 1e6 + total_out * 2.50 / 1e6
    logger.info("Estimated cost: $%.4f", cost)

    if dry_run:
        logger.info("Dry-run: skipping BigQuery insert.")
        return

    insert_notes(all_note_rows, all_label_rows)
    logger.info("Done. run_id=%s", run_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of patients (default: 50)")
    parser.add_argument("--strategy", choices=["both", "multilabel", "primary_dx"],
                        default="both")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip BigQuery insert (still writes JSONL backup)")
    args = parser.parse_args()

    strategies = (
        ["multilabel", "primary_dx"] if args.strategy == "both"
        else [args.strategy]
    )

    run(n_patients=args.limit, strategies=strategies, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
'''

Path("src/notes/run_day5.py").write_text(CONTENT)
print("Wrote src/notes/run_day5.py")
print(f"  {len(CONTENT.splitlines())} lines")