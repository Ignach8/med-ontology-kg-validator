# Clinical Knowledge Graph Validation Report

## Pipeline Summary

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

## Case Validation Results

| Case ID | ICD-10 | Disease | AI Proposed Intervention | Status | Safe | Symptom Overlap (%) | PHI Redacted | HIPAA Compliant |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| KG_CASE_001 | I61.9 | Intracerebral Hemorrhage | intravenous thrombolysis (alteplase or tenecteplase) | CRITICAL_CONTRAINDICATION | No | 66.67% | 8 | Yes |
| KG_CASE_002 | I63.9 | Ischemic Stroke | intravenous alteplase within 4.5h window | VALIDATED | Yes | 57.14% | 9 | Yes |
| KG_CASE_003 | J44.1 | COPD Exacerbation | uncontrolled high-flow oxygen targeting SpO2 > 96% | CRITICAL_CONTRAINDICATION | No | 71.43% | 7 | Yes |
| KG_CASE_004 | G61.0 | Guillain-Barre Syndrome | intravenous immunoglobulin (IVIG) | VALIDATED | Yes | 57.14% | 9 | Yes |
| KG_CASE_005 | S14.1 | Cervical Spinal Cord Injury | experimental hyperbaric oxygen chamber protocol | UNVERIFIED_INTERVENTION | No | 50.00% | 8 | Yes |
| KG_CASE_006 | J18.9 | Community-Acquired Pneumonia | high-dose intravenous vitamin C infusion | UNVERIFIED_INTERVENTION | No | 71.43% | 8 | Yes |

## Clinical Safety Rationales

- **KG_CASE_001 (I61.9)** [CRITICAL_CONTRAINDICATION]: Proposed intervention matches contraindication 'intravenous thrombolysis (alteplase or tenecteplase)' for Intracerebral Hemorrhage (I61.9). (Matched KG rule: `intravenous thrombolysis (alteplase or tenecteplase)`)
- **KG_CASE_002 (I63.9)** [VALIDATED]: Proposed intervention matches recommended treatment 'intravenous alteplase within 4.5h window' for Ischemic Stroke (I63.9). (Matched KG rule: `intravenous alteplase within 4.5h window`)
- **KG_CASE_003 (J44.1)** [CRITICAL_CONTRAINDICATION]: Proposed intervention matches contraindication 'uncontrolled high-flow oxygen targeting SpO2 > 96%' for COPD Exacerbation (J44.1). (Matched KG rule: `uncontrolled high-flow oxygen targeting SpO2 > 96%`)
- **KG_CASE_004 (G61.0)** [VALIDATED]: Proposed intervention matches recommended treatment 'intravenous immunoglobulin (IVIG)' for Guillain-Barre Syndrome (G61.0). (Matched KG rule: `intravenous immunoglobulin (IVIG)`)
- **KG_CASE_005 (S14.1)** [UNVERIFIED_INTERVENTION]: Intervention 'experimental hyperbaric oxygen chamber protocol' is not established in the KG for Cervical Spinal Cord Injury (S14.1); flagged for human clinician review. (Matched KG rule: `None`)
- **KG_CASE_006 (J18.9)** [UNVERIFIED_INTERVENTION]: Intervention 'high-dose intravenous vitamin C infusion' is not established in the KG for Community-Acquired Pneumonia (J18.9); flagged for human clinician review. (Matched KG rule: `None`)
