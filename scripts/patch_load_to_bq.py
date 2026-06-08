from pathlib import Path

path = Path("src/notes/load_to_bq.py")
src = path.read_text()

old = '    bigquery.SchemaField("primary_code", "STRING", mode="NULLABLE"),'
new = '    bigquery.SchemaField("primary_code", "STRING", mode="NULLABLE"),\n    bigquery.SchemaField("encounter_id", "STRING", mode="NULLABLE"),'

path.write_text(src.replace(old, new))
print("patched src/notes/load_to_bq.py")
