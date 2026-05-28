from pathlib import Path

CONTENT = '''"""Build LLM prompts from selected patient + condition dicts.

Two strategies:
  - multilabel:  full active problem list -> single multi-condition note
  - primary_dx:  one condition designated primary, others as comorbidities
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MULTILABEL_TEMPLATE = (PROMPTS_DIR / "multilabel.txt").read_text()
PRIMARY_DX_TEMPLATE = (PROMPTS_DIR / "primary_dx.txt").read_text()


def _format_condition_line(c: dict[str, Any]) -> str:
    date = c.get("start_date") or "unknown date"
    return f"- {c['icd10_code']} ({date}): {c['icd10_description']}"


def _conditions_block(conditions: list[dict[str, Any]]) -> str:
    return "\\n".join(_format_condition_line(c) for c in conditions)


def build_multilabel_prompt(patient: dict[str, Any]) -> str:
    """Single note covering the patient's full active problem list."""
    return MULTILABEL_TEMPLATE.format(
        conditions_block=_conditions_block(patient["conditions"])
    )


def _is_zcode(code: str) -> bool:
    return code.upper().startswith("Z")


def pick_primary(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Most recent non-Z, non-trivial condition. Falls back to most recent."""
    candidates = [
        c for c in conditions
        if not _is_zcode(c["icd10_code"]) and len(c["icd10_description"]) > 10
    ]
    return candidates[0] if candidates else conditions[0]


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
    """Smoke test: build both prompts for one real patient and print them."""
    import sys
    from src.notes.select_patients import fetch_patients

    patients = fetch_patients(n_patients=1)
    if not patients:
        print("No patients returned; check select_patients.")
        sys.exit(1)
    p = patients[0]

    print("=" * 70)
    print(f"PATIENT: {p['patient_id']}  ({len(p['conditions'])} conditions)")
    print("=" * 70)

    print("\\n--- MULTILABEL PROMPT ---\\n")
    print(build_multilabel_prompt(p))

    print("\\n--- PRIMARY-DX PROMPT ---\\n")
    prompt, primary = build_primary_dx_prompt(p)
    pc = primary["icd10_code"]
    pd = primary["icd10_description"]
    print(f"[picked primary: {pc} - {pd}]\\n")
    print(prompt)


if __name__ == "__main__":
    main()
'''

Path("src/notes/build_prompt.py").write_text(CONTENT)
print("Wrote src/notes/build_prompt.py")
print(f"  {len(CONTENT.splitlines())} lines")