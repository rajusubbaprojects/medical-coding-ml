-- Day 6: Clean-label views over notes_labels.
-- Two rollups: 3-char family (real target) and chapter (sanity check).
-- Both drop Z-codes (administrative / social determinants, not clinical Dx).

-- View 1: 3-char rollup (e.g. K06.8 -> K06, S86.122 -> S86)
CREATE OR REPLACE VIEW `medical-coding-ml-9848.medical_coding.notes_labels_clean_3char` AS
SELECT
  note_id,
  SUBSTR(icd10_code, 1, 3) AS label_code,
  MAX(is_primary) AS is_primary
FROM `medical-coding-ml-9848.medical_coding.notes_labels`
WHERE NOT STARTS_WITH(UPPER(icd10_code), 'Z')
GROUP BY note_id, label_code;

-- View 2: Chapter rollup (e.g. S86.122 -> S, K06.8 -> K)
-- "Chapter" here is loose -- ICD-10 has 22 official chapters with multi-letter
-- ranges (e.g. C00-D49 is Neoplasms). First-letter is a useful approximation
-- that's good enough for the sanity-check baseline.
CREATE OR REPLACE VIEW `medical-coding-ml-9848.medical_coding.notes_labels_clean_chapter` AS
SELECT
  note_id,
  SUBSTR(icd10_code, 1, 1) AS label_code,
  MAX(is_primary) AS is_primary
FROM `medical-coding-ml-9848.medical_coding.notes_labels`
WHERE NOT STARTS_WITH(UPPER(icd10_code), 'Z')
GROUP BY note_id, label_code;