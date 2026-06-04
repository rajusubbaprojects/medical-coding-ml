"""Generate synthetic discharge notes keyed on encounter_id.

Same pipeline as run_generation.py but uses select_encounters instead of
select_patients. Each encounter -> one note (multilabel strategy only,
since primary_dx makes less sense when conditions are already encounter-scoped).

Split key: encounter_id (stored in notes_synth.encounter_id field).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.notes.build_prompt import build_multilabel_prompt
from src.notes.generate import generate_note, MODEL
from src.notes.load_to_bq import ensure_tables, insert_notes
from src.notes.select_encounters import fetch_encounters

PROMPT_VERSION = "v1-encounter-multilabel"
BACKUP_DIR = Path("data/notes_synth")
INTER_CALL_SLEEP_SECONDS = 0.5

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_note_id(run_id: str, encounter_id: str) -> str:
    return f"{run_id}_{encounter_id}_encounter"


def run(*, n_encounters: int, dry_run: bool) -> None:
    run_id = make_run_id()
    logger.info("=== Encounter generation run %s ===", run_id)
    logger.info("Encounters: %d  Dry-run: %s", n_encounters, dry_run)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"run_{run_id}.jsonl"

    if not dry_run:
        ensure_tables()

    encounters = fetch_encounters(n_encounters=n_encounters)
    logger.info("Fetched %d encounters.", len(encounters))

    if len(encounters) < n_encounters:
        logger.warning("Wanted %d encounters but only found %d.", n_encounters, len(encounters))

    if n_encounters >= 200:
        est_cost = n_encounters * 0.002
        print(f"\n*** About to make {n_encounters} Gemini calls "
              f"(~{n_encounters * 7 / 60:.0f} min, ~${est_cost:.2f} estimated). "
              f"Continue? [y/N]")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    all_note_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    with open(backup_path, "a") as backup:
        for enc in tqdm(encounters, desc="generating"):
            try:
                prompt = build_multilabel_prompt(enc)
                gen = generate_note(prompt)

                note_id = make_note_id(run_id, enc["encounter_id"])
                labels = [c["icd10_code"] for c in enc["conditions"]]

                note_row = {
                    "note_id": note_id,
                    "run_id": run_id,
                    "patient_id": enc["patient_id"],
                    "strategy": "encounter-multilabel",
                    "prompt_version": PROMPT_VERSION,
                    "model": gen["model"],
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "note_text": gen["text"],
                    "labels": labels,
                    "primary_code": None,
                    "input_tokens": gen["input_tokens"],
                    "output_tokens": gen["output_tokens"],
                    "finish_reason": gen["finish_reason"],
                    "encounter_id": enc["encounter_id"],
                }

                label_rows = [
                    {
                        "note_id": note_id,
                        "icd10_code": c["icd10_code"],
                        "icd10_description": c["icd10_description"],
                        "is_primary": False,
                    }
                    for c in enc["conditions"]
                ]

                all_note_rows.append(note_row)
                all_label_rows.extend(label_rows)
                backup.write(json.dumps(note_row) + "\n")
                backup.flush()
                time.sleep(INTER_CALL_SLEEP_SECONDS)

            except Exception:
                logger.exception("Failed: encounter=%s", enc["encounter_id"])
                failures.append((enc["encounter_id"], enc["patient_id"]))

    logger.info("Generated: %d notes. Failures: %d", len(all_note_rows), len(failures))

    total_in = sum(r["input_tokens"] or 0 for r in all_note_rows)
    total_out = sum(r["output_tokens"] or 0 for r in all_note_rows)
    cost = total_in * 0.30 / 1e6 + total_out * 2.50 / 1e6
    logger.info("Tokens: %d in + %d out — estimated cost: $%.4f", total_in, total_out, cost)
    logger.info("Backup: %s", backup_path)

    if dry_run:
        logger.info("Dry-run: skipping BigQuery insert.")
        if all_note_rows:
            print("\nSample note (dry-run):")
            print(all_note_rows[0]["note_text"][:400])
        return

    insert_notes(all_note_rows, all_label_rows)
    logger.info("Done. run_id=%s", run_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of encounters to generate notes for")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(n_encounters=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
