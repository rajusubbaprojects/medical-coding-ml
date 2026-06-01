"""Build LLM prompts from selected patient + condition dicts.

Two strategies:
  - multilabel:  full active problem list -> single multi-condition note
  - primary_dx:  one condition designated primary, others as comorbidities

Day 8 update: smarter primary-Dx picker (chapter scoring + trivial-word penalty)
replaces the naive "first non-Z, non-trivial" heuristic from Day 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MULTILABEL_TEMPLATE = (PROMPTS_DIR / "multilabel.txt").read_text()
PRIMARY_DX_TEMPLATE = (PROMPTS_DIR / "primary_dx.txt").read_text()


# ---------------------------------------------------------------------------
# Primary-Dx scoring
# ---------------------------------------------------------------------------

# ICD-10 chapter weights: how likely is a code from this chapter to be an
# admission's primary diagnosis? Higher = more likely.
# Keys are tuples (chapter_letter, min_2nd_char, max_2nd_char) where the range
# is inclusive. None for min/max means unbounded.
_CHAPTER_WEIGHTS: list[tuple[str, str, str, int]] = [
    # (letter, 2nd-char-min, 2nd-char-max, weight)
    ("I", "0", "9", 4),   # Cardiovascular
    ("J", "0", "9", 4),   # Respiratory
    ("K", "2", "9", 4),   # GI (excluding K0x dental)
    ("K", "0", "1", 0),   # K0x: oral/dental — rarely admission cause
    ("S", "0", "9", 4),   # Injury
    ("T", "0", "9", 4),   # Poisoning / injury sequelae
    ("A", "0", "9", 3),   # Infection
    ("B", "0", "9", 3),   # Infection
    ("C", "0", "9", 3),   # Neoplasm
    ("D", "0", "4", 3),   # Neoplasms (in situ etc)
    ("D", "5", "9", 3),   # Blood disorders (anemia etc)
    ("E", "0", "9", 3),   # Endocrine
    ("N", "0", "9", 3),   # Renal/GU
    ("O", "0", "9", 2),   # Pregnancy
    ("R", "0", "9", 2),   # Symptoms / abnormal findings
    ("G", "0", "9", 2),   # Nervous system
    ("H", "0", "9", 1),   # Eye/ear (H6x otitis is OK; rest is chronic)
    ("L", "0", "9", 1),   # Skin
    ("M", "0", "9", 1),   # MSK (mostly chronic)
    ("F", "0", "9", 1),   # Mental health
    ("Z", "0", "9", 0),   # Admin/social (also filtered by Z-rule below)
]

_TRIVIAL_PHRASES = (
    "other specified",
    "other disorders",
    "unspecified",
    "edentulous",
    "in remission",
    "history of",
    "personal history",
    "screening for",
)


def _chapter_weight(code: str) -> int:
    """Return chapter weight from the lookup table. Default 1 for unmatched."""
    if not code:
        return 1
    letter = code[0].upper()
    second = code[1] if len(code) > 1 else "0"
    for ch_letter, lo, hi, w in _CHAPTER_WEIGHTS:
        if letter == ch_letter and lo <= second <= hi:
            return w
    return 1  # default for unmatched chapters


def _is_zcode(code: str) -> bool:
    return code.upper().startswith("Z")


def _trivial_penalty(description: str) -> int:
    desc = description.lower()
    hits = sum(1 for phrase in _TRIVIAL_PHRASES if phrase in desc)
    return -2 * hits


def _specificity_bonus(code: str) -> int:
    # Codes longer than 4 chars (e.g. S86.122) are more specific than 3-char (S86).
    return 1 if len(code) > 4 else 0


def score_for_primary(condition: dict[str, Any]) -> int:
    """Higher score = better candidate for primary admission diagnosis."""
    code = condition["icd10_code"]
    if _is_zcode(code):
        return -100  # never pick Z-codes
    return (
        _chapter_weight(code)
        + _trivial_penalty(condition["icd10_description"])
        + _specificity_bonus(code)
    )


def pick_primary(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best primary-Dx candidate via chapter-weighted scoring.

    Tie-break: most recent by start_date.
    Falls back to first condition if everything scores <= 0 (shouldn't happen
    in practice unless all conditions are Z-codes, in which case the caller
    should have filtered earlier).
    """
    scored = sorted(
        conditions,
        key=lambda c: (
            -score_for_primary(c),
            c["start_date"] or "0000-00-00",  # most recent first on tie
        ),
        reverse=False,  # we want most NEGATIVE first since we negated score
    )
    # The sort above isn't quite right — let's do it more readably:
    best = max(
        conditions,
        key=lambda c: (
            score_for_primary(c),
            c["start_date"] or "0000-00-00",
        ),
    )
    return best


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_condition_line(c: dict[str, Any]) -> str:
    date = c.get("start_date") or "unknown date"
    return f"- {c['icd10_code']} ({date}): {c['icd10_description']}"


def _conditions_block(conditions: list[dict[str, Any]]) -> str:
    return "\n".join(_format_condition_line(c) for c in conditions)


def build_multilabel_prompt(patient: dict[str, Any]) -> str:
    """Single note covering the patient's full active problem list."""
    return MULTILABEL_TEMPLATE.format(
        conditions_block=_conditions_block(patient["conditions"])
    )


def build_primary_dx_prompt(patient: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Returns (prompt_string, primary_condition_dict)."""
    primary = pick_primary(patient["conditions"])
    secondary = [c for c in patient["conditions"] if c is not primary]

    primary_code = primary["icd10_code"]
    primary_desc = primary["icd10_description"]
    primary_str = f"{primary_desc} (ICD-10: {primary_code})"
    secondary_block = _conditions_block(secondary) if secondary else "- (none)"

    prompt = PRIMARY_DX_TEMPLATE.format(
        primary_condition=primary_str,
        secondary_block=secondary_block,
    )
    return prompt, primary


def main() -> None:
    """Smoke test + diagnostic: print primary-Dx picks for the first 10 patients."""
    from src.notes.select_patients import fetch_patients

    patients = fetch_patients(n_patients=10)
    print(f"Primary-Dx picks for {len(patients)} patients (new picker):")
    print(f"  {'patient_id (head)':18s} {'code':10s} {'score':>5s}  description")
    for p in patients:
        primary = pick_primary(p["conditions"])
        s = score_for_primary(primary)
        pid_head = p["patient_id"][:8]
        desc = primary["icd10_description"][:50]
        print(f"  {pid_head:18s} {primary['icd10_code']:10s} {s:>5d}  {desc}")


if __name__ == "__main__":
    main()
