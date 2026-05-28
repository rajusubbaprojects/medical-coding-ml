"""Print 3 random notes from the latest run for eyeball QA."""

from google.cloud import bigquery

PROJECT = "medical-coding-ml-9848"

client = bigquery.Client(project=PROJECT)

query = """
WITH latest AS (
  SELECT MAX(run_id) AS run_id
  FROM `medical-coding-ml-9848.medical_coding.notes_synth`
)
SELECT
  n.note_id,
  n.strategy,
  n.primary_code,
  n.labels,
  n.note_text,
  n.output_tokens
FROM `medical-coding-ml-9848.medical_coding.notes_synth` n
JOIN latest USING (run_id)
ORDER BY RAND()
LIMIT 3
"""

for i, row in enumerate(client.query(query).result(), 1):
    print("=" * 78)
    print(f"SAMPLE {i}/3")
    print(f"  note_id:      {row.note_id}")
    print(f"  strategy:     {row.strategy}")
    print(f"  primary:      {row.primary_code}")
    print(f"  labels ({len(row.labels)}): {', '.join(row.labels[:8])}"
          f"{' ...' if len(row.labels) > 8 else ''}")
    print(f"  output_tokens: {row.output_tokens}")
    print("-" * 78)
    print(row.note_text)
    print()