from pathlib import Path

MULTILABEL = """You are a board-certified hospitalist writing a discharge summary for a patient who has just completed an inpatient stay. Produce a realistic, clinically plausible discharge note that addresses the full active problem list.

PATIENT CONDITIONS (from the medical record, most recent first):
{conditions_block}

INSTRUCTIONS:
- Write in standard discharge-summary format with these sections, in this order:
  HOSPITAL COURSE
  ACTIVE PROBLEMS / ASSESSMENT AND PLAN
  DISCHARGE MEDICATIONS
  FOLLOW-UP
- Address every condition listed above somewhere in the note. Group related conditions (e.g., diabetes and its complications) where clinically appropriate.
- Use natural clinical prose. Do NOT write a bulleted code list. NEVER write ICD-10 codes anywhere in the note - no codes in parentheses, no codes in section headers, no codes at all. Refer to conditions by their clinical names only.
- Mention plausible labs, vitals, medications, and follow-up arrangements consistent with the conditions. You may invent specific values (e.g., "A1c 8.2%") so long as they are consistent with the diagnoses.
- For social/behavioral codes (Z-codes such as unemployment, housing instability), incorporate them briefly into a SOCIAL HISTORY mention or the discharge plan, not as primary diagnoses.
- Length: roughly 300-500 words. Do not include a header with patient name, MRN, or dates - start directly with "HOSPITAL COURSE".
- Do not include any preamble, explanation, or markdown formatting. Output only the note text.

BEGIN DISCHARGE SUMMARY:
"""

PRIMARY_DX = """You are a board-certified hospitalist writing a discharge summary for a patient whose admission was primarily for: {primary_condition}.

The patient also has the following relevant past medical history and active problems:
{secondary_block}

INSTRUCTIONS:
- Write a focused discharge summary centered on the primary diagnosis. The note should make clear why the patient was admitted and what was done.
- Use standard discharge-summary format with these sections, in this order:
  HOSPITAL COURSE
  ASSESSMENT AND PLAN
  DISCHARGE MEDICATIONS
  FOLLOW-UP
- The primary diagnosis should dominate the HOSPITAL COURSE. Secondary problems should appear in ASSESSMENT AND PLAN as comorbidities being managed concurrently.
- Use natural clinical prose. Do NOT write a bulleted code list. NEVER write ICD-10 codes anywhere in the note - no codes in parentheses, no codes in section headers, no codes at all. Refer to conditions by their clinical names only.
- Mention plausible labs, vitals, medications, and follow-up arrangements consistent with the primary diagnosis and comorbidities. You may invent specific values consistent with the diagnoses.
- Length: roughly 250-400 words. Do not include a header with patient name, MRN, or dates - start directly with "HOSPITAL COURSE".
- Do not include any preamble, explanation, or markdown formatting. Output only the note text.

BEGIN DISCHARGE SUMMARY:
"""

Path("prompts/multilabel.txt").write_text(MULTILABEL)
Path("prompts/primary_dx.txt").write_text(PRIMARY_DX)
print("Wrote prompts/multilabel.txt and prompts/primary_dx.txt")