# med-ontology-kg-validator

## Overview and Clinical Motivation

Large language models deployed in clinical decision support can propose interventions that are unverified or directly contraindicated for a patient's diagnosis (for example, recommending intravenous thrombolysis in intracerebral hemorrhage or uncontrolled high-flow oxygen in hypercapnic COPD exacerbation). In parallel, sending raw clinical narratives into downstream analytics or model evaluation pipelines introduces protected health information (PHI) exposure risks.

`med-ontology-kg-validator` is a deterministic Python 3.11 validation framework that combines:

- **Standardized Clinical Ontologies**: Encodes neurology, neurorehabilitation, and respiratory conditions mapped to **ICD-10**, **SNOMED CT**, and **LOINC** laboratory/vital codes inside a directed **NetworkX** Knowledge Graph.
- **HIPAA Safe Harbor De-identification**: Applies regex-based PHI detection and post-redaction compliance verification (aligned with 45 CFR 164.514(b)(2) identifiers) to strip names, dates, SSNs, Chilean RUT/RUN identifiers, phone numbers, email addresses, and MRNs while preserving clinical codes and physiological ratios.
- **Safety-First Recommendation Auditing**: Evaluates AI-proposed interventions against explicit `CONTRAINDICATED_WITH` and `TREATED_BY` graph relations, classifying outputs as `CRITICAL_CONTRAINDICATION`, `VALIDATED`, `UNVERIFIED_INTERVENTION` (flagged for human clinician review), or `UNKNOWN_CONDITION`, and quantifies observed symptom alignment against `HAS_SYMPTOM` expectations.

## Architecture and Workflow

```text
+-----------------------------------+      +------------------------------------+
| ontologies/clinical_ontologies.json|      | data/sample_clinical_cases.json    |
| (ICD-10, SNOMED CT, LOINC)        |      | (Synthetic notes + AI proposals)   |
+-----------------+-----------------+      +-----------------+------------------+
                  |                                          |
                  v                                          v
    +---------------------------+              +----------------------------+
    | src/kg_builder.py         |              | src/hipaa_deidentifier.py  |
    | - NetworkX DiGraph        |              | - Safe Harbor regex rules  |
    | - Typed nodes & relations |              | - Residual PHI compliance  |
    | - PyVis HTML exporter     |              |   verification             |
    +-------------+-------------+              +-------------+--------------+
                  |                                          |
                  +--------------------+---------------------+
                                       |
                                       v
                         +---------------------------+
                         | src/validator.py          |
                         | KGValidator (Pydantic v2) |
                         | - Contraindication check  |
                         | - Treatment validation    |
                         | - Symptom cluster overlap |
                         +-------------+-------------+
                                       |
                                       v
      +----------------------------------------------------------------+
      | main.py -> results/                                            |
      | - results/clinical_kg.html        (Interactive PyVis graph)    |
      | - results/validation_results.json (De-identified audit output) |
      | - results/validation_report.md    (Markdown summary table)     |
      +----------------------------------------------------------------+
```

## Project Structure

```text
med-ontology-kg-validator/
|-- data/
|   `-- sample_clinical_cases.json    # 6 synthetic clinical cases with fake PHI
|-- ontologies/
|   `-- clinical_ontologies.json      # 8 diseases (ICD-10/SNOMED CT) + 10 LOINC labs
|-- results/
|   |-- clinical_kg.html              # Interactive PyVis Knowledge Graph
|   |-- validation_results.json       # Structured Pydantic v2 JSON output
|   `-- validation_report.md          # Markdown validation summary
|-- src/
|   |-- __init__.py
|   |-- hipaa_deidentifier.py         # Regex + Pydantic v2 Safe Harbor redactor
|   |-- kg_builder.py                 # NetworkX DiGraph builder & PyVis exporter
|   `-- validator.py                  # KGValidator recommendation & symptom checks
|-- tests/
|   |-- __init__.py
|   `-- test_pipeline.py              # Pytest unit and integration test suite
|-- main.py                           # End-to-end execution entrypoint
|-- requirements.txt
`-- README.md
```

## Quickstart

### 1. Install Dependencies

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

```bash
python main.py
```

This builds the clinical Knowledge Graph, de-identifies all clinical notes, runs ontology validation across all cases, and writes outputs to `results/`.

### 3. Run the Test Suite

```bash
pytest -v
```

### 4. Inspect the Interactive Knowledge Graph

After running `python main.py`, open `results/clinical_kg.html` directly in any modern web browser (for example, double-click the file, run `start results/clinical_kg.html` on Windows, `open results/clinical_kg.html` on macOS, or `xdg-open results/clinical_kg.html` on Linux).

The PyVis visualization uses Barnes-Hut physics layout with color-coded node types and hover tooltips displaying ICD-10, SNOMED CT, and LOINC metadata:

- Blue (`disease`): ICD-10 / SNOMED CT clinical conditions
- Amber (`symptom`): Expected clinical manifestations (`HAS_SYMPTOM`)
- Green (`intervention`): Guideline-aligned treatments (`TREATED_BY`)
- Red (`contraindication`): Contraindicated actions (`CONTRAINDICATED_WITH`, dashed edges)
- Violet diamond (`lab_code`): LOINC laboratory and vital sign references

## Sample Validation Results

Metrics from executing `python main.py` against `data/sample_clinical_cases.json` (sourced directly from `results/validation_report.md`):

| Metric | Value |
| --- | --- |
| Knowledge Graph Nodes | 153 |
| Knowledge Graph Edges | 141 |
| Clinical Cases Evaluated | 6 |
| Total PHI Entities Redacted | 49 |
| All Notes HIPAA Compliant | Yes |
| Validated Interventions | 2 |
| Critical Contraindications | 2 |
| Unverified Interventions (Clinician Review) | 2 |
| Unknown Conditions | 0 |
| Mean Symptom Cluster Overlap (%) | 62.30% |

| Case ID | ICD-10 | Disease | AI Proposed Intervention | Status | Safe | Symptom Overlap (%) | PHI Redacted | HIPAA Compliant |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| KG_CASE_001 | I61.9 | Intracerebral Hemorrhage | intravenous thrombolysis (alteplase or tenecteplase) | CRITICAL_CONTRAINDICATION | No | 66.67% | 8 | Yes |
| KG_CASE_002 | I63.9 | Ischemic Stroke | intravenous alteplase within 4.5h window | VALIDATED | Yes | 57.14% | 9 | Yes |
| KG_CASE_003 | J44.1 | COPD Exacerbation | uncontrolled high-flow oxygen targeting SpO2 > 96% | CRITICAL_CONTRAINDICATION | No | 71.43% | 7 | Yes |
| KG_CASE_004 | G61.0 | Guillain-Barre Syndrome | intravenous immunoglobulin (IVIG) | VALIDATED | Yes | 57.14% | 9 | Yes |
| KG_CASE_005 | S14.1 | Cervical Spinal Cord Injury | experimental hyperbaric oxygen chamber protocol | UNVERIFIED_INTERVENTION | No | 50.00% | 8 | Yes |
| KG_CASE_006 | J18.9 | Community-Acquired Pneumonia | high-dose intravenous vitamin C infusion | UNVERIFIED_INTERVENTION | No | 71.43% | 8 | Yes |

## Synthetic Data and Compliance Notice

All clinical vignettes and embedded identifiers in `data/sample_clinical_cases.json` (names, dates, SSNs, Chilean RUT/RUN numbers, phone numbers, email addresses, and medical record numbers) are **100% synthetic** and fabricated solely for HIPAA Safe Harbor de-identification and ontology validation testing. No real patient data is included in this repository, and raw clinical notes are never persisted into `results/validation_results.json` or `results/validation_report.md`.
