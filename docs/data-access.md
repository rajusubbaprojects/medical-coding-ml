# Data access: MIMIC-IV via PhysioNet

This project trains on MIMIC-IV-Note, a de-identified clinical notes dataset from the Beth Israel Deaconess Medical Center, released via PhysioNet. You cannot use this data without:

1. A PhysioNet account
2. Completed CITI "Data or Specimens Only Research" training
3. Signed Data Use Agreement
4. Approved credentialed access to MIMIC-IV

**This takes 3–10 days end to end.** Start now.

## Step 1: PhysioNet account

Register at https://physionet.org/register/. Use a real name. An institutional email helps but isn't required.

## Step 2: CITI training (4–6 hours)

From your PhysioNet profile, follow the "Training" tab. Complete the modules under "Data or Specimens Only Research". Pace yourself across a few evenings — it's mostly multiple choice with reading material.

When finished, download the completion certificate and upload it to your PhysioNet profile.

## Step 3: Request MIMIC-IV access

Go to https://physionet.org/content/mimiciv/ and click "Request access". Sign the DUA. Approval is usually 1–3 days.

Then go to https://physionet.org/content/mimic-iv-note/ and request access to the notes specifically. Same DUA, same approval window.

## Step 4: Download to GCS (not your laptop)

Once approved, **do not download MIMIC to your laptop**. Stream it directly from PhysioNet to your private GCS bucket:

```bash
# From a Cloud Shell or a Compute Engine VM in your project
export PHYSIONET_USER=your_username
gsutil -m cp -r \
  "https://${PHYSIONET_USER}@physionet.org/files/mimic-iv-note/2.2/note/" \
  gs://${PROJECT_ID}-data/mimic-iv-note/
```

The bucket has `public_access_prevention = "enforced"` so this data is never reachable from outside the project.

## What never leaves the controlled environment

- Raw note text
- Subject IDs, hadm IDs, dates
- Any aggregation that could re-identify a patient (e.g., outlier counts in a small cohort)

## What can go in the public repo

- Code
- Aggregated metrics (precision/recall over the full eval set)
- Confusion matrices over code categories (not individual patients)
- Architectural diagrams
- Synthetic example inputs you write yourself for blog post screenshots

If you're not 100% sure something is safe to publish, the answer is don't.

## Status checklist

Track your progress through the approval workflow:

- [ ] PhysioNet account created
- [ ] CITI training started
- [ ] CITI training completed
- [ ] CITI certificate uploaded to PhysioNet
- [ ] MIMIC-IV access requested
- [ ] MIMIC-IV access approved
- [ ] MIMIC-IV-Note access requested
- [ ] MIMIC-IV-Note access approved
- [ ] Data downloaded to GCS bucket
- [ ] First successful BigQuery query against the data