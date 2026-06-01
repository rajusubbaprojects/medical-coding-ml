from pathlib import Path

CONTENT = '''"""Generate synthetic discharge notes end-to-end.

Originally Day 5's run_day5.py; renamed Day 8 since it's infrastructure
that gets re-used across many runs (Day 5 first batch, Day 8a regeneration,
Day 8b scale-up, etc).

Pipeline:
    1. fetch_patients          -> list[patient_dict]
       (filtered by --patients-from-run / --exclude-patients-from-run if given)
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
import time
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from tqdm import tqdm

from src.notes.build_prompt import build_multilabel_prompt, build_primary_dx_prompt
from src.notes.generate import generate_note, MODEL
from src.notes.load_to_bq import ensure_tables, insert_notes, PROJECT, DATASET, NOTES_TABLE
from src.notes.select_patients import fetch_patients

PROMPT_VERSION_MULTI = "v1-multilabel"
PROMPT_VERSION_PRIMARY = "v2-primary_dx"  # bumped on Day 8: new picker

BACKUP_DIR = Path("data/notes_synth")
INTER_CALL_SLEEP_SECONDS = 0.5  # Belt-and-suspenders against rate limits

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_note_id(run_id: str, patient_id: str, strategy: str) -> str:
    return f"{run_id}_{patient_id}_{strategy}"


def _patients_in_run(run_id: str) -> set[str]:
    """Return set of patient_ids that have notes from a given run_id."""
    client = bigquery.Client(project=PROJECT)
    query = f"""
        SELECT DISTINCT patient_id
        FROM `{PROJECT}.{DATASET}.{NOTES_TABLE}`
        WHERE run_id = @run_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return {row.patient_id for row in rows}


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


def run(
    *,
    n_patients: int,
    strategies: list[str],
    dry_run: bool,
    patients_from_run: str | None,
    exclude_patients_from_run: str | None,
) -> None:
    run_id = make_run_id()
    logger.info("=== Run %s ===", run_id)
    logger.info("Patients: %d  Strategies: %s  Dry-run: %s",
                n_patients, strategies, dry_run)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"run_{run_id}.jsonl"

    if not dry_run:
        ensure_tables()

    # Patient filtering
    include_pids: set[str] | None = None
    exclude_pids: set[str] = set()
    if patients_from_run:
        include_pids = _patients_in_run(patients_from_run)
        logger.info("Limiting to %d patients from run %s",
                    len(include_pids), patients_from_run)
    if exclude_patients_from_run:
        exclude_pids = _patients_in_run(exclude_patients_from_run)
        logger.info("Excluding %d patients from run %s",
                    len(exclude_pids), exclude_patients_from_run)

    # Fetch a generous pool, then apply filters
    pool_size = max(n_patients * 3, 200) if exclude_pids else n_patients * 2
    if include_pids:
        # Pull broad pool then intersect
        all_patients = fetch_patients(n_patients=max(len(include_pids) * 2, n_patients))
        patients = [p for p in all_patients if p["patient_id"] in include_pids]
        patients = patients[:n_patients]
    else:
        all_patients = fetch_patients(n_patients=pool_size)
        patients = [p for p in all_patients if p["patient_id"] not in exclude_pids]
        patients = patients[:n_patients]

    logger.info("Selected %d patients after filtering.", len(patients))
    if len(patients) < n_patients:
        logger.warning("Wanted %d patients but only found %d. Continuing.",
                       n_patients, len(patients))

    all_note_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    total_calls = len(patients) * len(strategies)

    # Cost preview & confirmation for large runs
    if total_calls >= 200:
        est_cost = total_calls * 0.002  # ~$0.002/note based on Day 5 actuals
        print(f"\\n*** About to make {total_calls} Gemini calls "
              f"(~{total_calls * 7 / 60:.0f} min, ~${est_cost:.2f} estimated). "
              f"Continue? [y/N]")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

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

                    time.sleep(INTER_CALL_SLEEP_SECONDS)
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
    parser.add_argument("--patients-from-run", default=None,
                        help="Use the same patients as a previous run_id")
    parser.add_argument("--exclude-patients-from-run", default=None,
                        help="Skip patients used in a previous run_id")
    args = parser.parse_args()

    strategies = (
        ["multilabel", "primary_dx"] if args.strategy == "both"
        else [args.strategy]
    )

    run(
        n_patients=args.limit,
        strategies=strategies,
        dry_run=args.dry_run,
        patients_from_run=args.patients_from_run,
        exclude_patients_from_run=args.exclude_patients_from_run,
    )


if __name__ == "__main__":
    main()
'''

Path("src/notes/run_generation.py").write_text(CONTENT)
print("Wrote src/notes/run_generation.py")
print(f"  {len(CONTENT.splitlines())} lines")