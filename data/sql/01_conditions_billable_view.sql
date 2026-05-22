-- conditions_billable: Synthea conditions with their ICD-10-CM mappings.
--
-- Joins the Synthea conditions table (SNOMED-coded) to the OHDSI vocabulary
-- tables to produce a row per condition with both source SNOMED and target
-- ICD-10-CM codes.
--
-- Rows where the SNOMED code has no ICD-10-CM mapping are dropped — these are
-- typically administrative codes (e.g. "Medication review due") and social
-- determinants of health (e.g. "Full-time employment") that hospitals don't
-- bill on. This filtering mirrors what real coders do.
--
-- Run with:
--   bq query --use_legacy_sql=false < data/sql/01_conditions_billable_view.sql

CREATE OR REPLACE VIEW `medical-coding-ml-9848.medical_coding.conditions_billable` AS

WITH
  -- Step 1: every SNOMED concept in our system, indexed by the text code
  -- Synthea writes ("44054006", "162864005", etc.).
  snomed_concepts AS (
    SELECT
      concept_id      AS snomed_concept_id,
      concept_code    AS snomed_code,
      concept_name    AS snomed_name
    FROM `medical-coding-ml-9848.medical_coding.concept`
    WHERE vocabulary_id = 'SNOMED'
      AND standard_concept = 'S'  -- only "standard" concepts; drops deprecated entries
  ),

  -- Step 2: every ICD-10-CM concept.
  icd10_concepts AS (
    SELECT
      concept_id      AS icd10_concept_id,
      concept_code    AS icd10_code,
      concept_name    AS icd10_name
    FROM `medical-coding-ml-9848.medical_coding.concept`
    WHERE vocabulary_id = 'ICD10CM'
  ),

  -- Step 3: SNOMED → ICD-10 mappings. "Maps to" is the OHDSI relationship_id
  -- specifically used for code translation between vocabularies.
  -- One SNOMED concept can map to multiple ICD-10 concepts (specificity).
  snomed_to_icd10 AS (
    SELECT
      concept_id_1 AS snomed_concept_id,
      concept_id_2 AS icd10_concept_id
    FROM `medical-coding-ml-9848.medical_coding.concept_relationship`
    WHERE relationship_id = 'Mapped from'
  )

-- Final join: stitch Synthea conditions → SNOMED concept → mapping → ICD-10 concept.
SELECT
  c.PATIENT       AS patient_id,
  c.ENCOUNTER     AS encounter_id,
  c.START         AS condition_start_date,
  c.STOP          AS condition_stop_date,
  c.CODE          AS snomed_code,
  c.DESCRIPTION   AS snomed_description,
  icd.icd10_code,
  icd.icd10_name  AS icd10_description
FROM `medical-coding-ml-9848.medical_coding.conditions` AS c
INNER JOIN snomed_concepts AS sn  ON CAST(c.CODE AS STRING) = sn.snomed_code
INNER JOIN snomed_to_icd10 AS m   ON sn.snomed_concept_id = m.snomed_concept_id
INNER JOIN icd10_concepts  AS icd ON m.icd10_concept_id = icd.icd10_concept_id;