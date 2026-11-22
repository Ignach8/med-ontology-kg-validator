"""Unit and integration tests for the clinical KG validation pipeline."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from main import PipelineReport, run_pipeline
from src.hipaa_deidentifier import (
    DeidentificationResult,
    DeidentifierConfig,
    HIPAADeidentifier,
    PHICategory,
    deidentify_text,
)
from src.kg_builder import EdgeRelation, NodeType, build_knowledge_graph
from src.validator import (
    KGValidator,
    SymptomClusterResult,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture(scope="module")
def clinical_graph() -> nx.DiGraph:
    return build_knowledge_graph()


@pytest.fixture(scope="module")
def validator(clinical_graph: nx.DiGraph) -> KGValidator:
    return KGValidator(clinical_graph)


def test_hipaa_deidentifier_redacts_all_phi_categories() -> None:
    raw_note: str = (
        "Patient: Elena Morales (MRN: 84219034, DOB: 1957-06-18, RUT: 12.345.678-K) "
        "and SSN: 123-45-6789 was admitted on 03/14/2026 and reviewed on 25/04/2026 "
        "and November 21, 2026 by Dr. Marcus Vance. Ms. Elena Morales and "
        "Mr. Arthur Pendelton provided phones (555) 234-8910 and +56 9 7123 4567 "
        "and email m.vance@neuroclinic.org. Vitals: BP 120/80 mmHg, SpO2 94% "
        "(LOINC 59408-5), ICD-10 I63.9, SNOMED 422504002."
    )

    result: DeidentificationResult = HIPAADeidentifier().deidentify(raw_note)

    assert result.original_length == len(raw_note)
    assert result.is_hipaa_compliant is True

    for phi_value in (
        "Elena Morales",
        "Marcus Vance",
        "Arthur Pendelton",
        "84219034",
        "1957-06-18",
        "03/14/2026",
        "25/04/2026",
        "November 21, 2026",
        "12.345.678-K",
        "123-45-6789",
        "(555) 234-8910",
        "+56 9 7123 4567",
        "m.vance@neuroclinic.org",
    ):
        assert phi_value not in result.redacted_text

    for placeholder in ("[NAME]", "[DATE]", "[ID_REDACTED]", "[PHONE]", "[EMAIL]", "[MRN]"):
        assert placeholder in result.redacted_text

    assert result.entities_redacted[PHICategory.NAME.value] == 4
    assert result.entities_redacted[PHICategory.DATE.value] == 4
    assert result.entities_redacted[PHICategory.NATIONAL_ID.value] == 2
    assert result.entities_redacted[PHICategory.PHONE.value] == 2
    assert result.entities_redacted[PHICategory.EMAIL.value] == 1
    assert result.entities_redacted[PHICategory.MRN.value] == 1

    # Clinical identifiers and physiological ratios must remain intact.
    for clinical_token in ("120/80", "59408-5", "I63.9", "422504002"):
        assert clinical_token in result.redacted_text


def test_hipaa_deidentifier_detects_residual_phi_when_rule_disabled() -> None:
    raw_note: str = "Patient: John Doe, DOB: 1980-05-12, MRN: 123456."
    result: DeidentificationResult = deidentify_text(
        raw_note,
        config=DeidentifierConfig(redact_dates=False),
    )

    assert "1980-05-12" in result.redacted_text
    assert "John Doe" not in result.redacted_text
    assert result.is_hipaa_compliant is False


@pytest.mark.parametrize(
    ("icd10_code", "intervention"),
    [
        ("I61.9", "intravenous thrombolysis (alteplase or tenecteplase)"),
        ("I61.9", "Intravenous Thrombolysis"),
        ("I61.9", "alteplase"),
        ("J44.1", "uncontrolled high-flow oxygen targeting SpO2 > 96%"),
        ("J44.1", "high-flow oxygen"),
        ("G61.0", "succinylcholine"),
    ],
)
def test_validator_flags_critical_contraindication(
    validator: KGValidator,
    icd10_code: str,
    intervention: str,
) -> None:
    result: ValidationResult = validator.validate_recommendation(
        icd10_code=icd10_code,
        proposed_intervention=intervention,
    )

    assert result.status == ValidationStatus.CRITICAL_CONTRAINDICATION
    assert result.is_safe is False
    assert result.matched_rule is not None
    assert result.requires_clinician_review is True


@pytest.mark.parametrize(
    ("icd10_code", "intervention"),
    [
        ("I63.9", "intravenous alteplase within 4.5h window"),
        ("I63.9", "Non-Contrast Head CT"),
        ("G61.0", "intravenous immunoglobulin (IVIG)"),
        ("G61.0", "IVIG"),
        ("J44.1", "systemic corticosteroids"),
    ],
)
def test_validator_validates_expected_intervention(
    validator: KGValidator,
    icd10_code: str,
    intervention: str,
) -> None:
    result: ValidationResult = validator.validate_recommendation(
        icd10_code=icd10_code,
        proposed_intervention=intervention,
    )

    assert result.status == ValidationStatus.VALIDATED
    assert result.is_safe is True
    assert result.matched_rule is not None
    assert result.requires_clinician_review is False


@pytest.mark.parametrize(
    ("icd10_code", "intervention"),
    [
        ("S14.1", "experimental hyperbaric oxygen chamber protocol"),
        ("J18.9", "high-dose intravenous vitamin C infusion"),
        ("I63.9", "unlisted herbal protocol"),
    ],
)
def test_validator_flags_unverified_intervention(
    validator: KGValidator,
    icd10_code: str,
    intervention: str,
) -> None:
    result: ValidationResult = validator.validate_recommendation(
        icd10_code=icd10_code,
        proposed_intervention=intervention,
    )

    assert result.status == ValidationStatus.UNVERIFIED_INTERVENTION
    assert result.is_safe is False
    assert result.matched_rule is None
    assert result.requires_clinician_review is True


def test_validator_flags_unknown_icd10_code(validator: KGValidator) -> None:
    result: ValidationResult = validator.validate_recommendation(
        icd10_code="Z99.9",
        proposed_intervention="supplemental oxygen therapy",
    )

    assert result.status == ValidationStatus.UNKNOWN_CONDITION
    assert result.is_safe is False
    assert result.disease_name is None
    assert result.requires_clinician_review is True


def test_validate_symptom_cluster_overlap_percentage(validator: KGValidator) -> None:
    # I63.9 defines 7 expected symptoms; 3 matching symptoms yield 42.86%.
    partial: SymptomClusterResult = validator.validate_symptom_cluster(
        icd10_code="I63.9",
        observed_symptoms=["Facial Droop", "aphasia", "dysphagia"],
    )
    assert partial.condition_found is True
    assert partial.overlap_score == round(3 / 7 * 100.0, 2)
    assert partial.observed_match_pct == 100.0
    assert len(partial.matched_symptoms) == 3
    assert len(partial.missing_expected_symptoms) == 4

    # Duplicate observed symptoms must not inflate overlap; unmatched symptoms are tracked.
    mixed: SymptomClusterResult = validator.validate_symptom_cluster(
        icd10_code="S14.1",
        observed_symptoms=["tetraplegia", "bradycardia", "bradycardia", "skin rash"],
    )
    assert mixed.condition_found is True
    assert mixed.overlap_score == round(2 / 8 * 100.0, 2)
    assert mixed.unmatched_observed_symptoms == ["skin rash"]

    # Full match on all 6 expected symptoms of I61.9 yields 100.0%.
    full: SymptomClusterResult = validator.validate_symptom_cluster(
        icd10_code="I61.9",
        observed_symptoms=[
            "sudden severe headache",
            "progressive focal neurological deficit",
            "decreased level of consciousness",
            "nausea and vomiting",
            "severe hypertension",
            "seizures",
        ],
    )
    assert full.overlap_score == 100.0
    assert full.missing_expected_symptoms == []

    # Empty symptoms and unknown conditions return 0.0%.
    assert validator.validate_symptom_cluster("I63.9", []).overlap_score == 0.0
    unknown: SymptomClusterResult = validator.validate_symptom_cluster("Z99.9", ["fever"])
    assert unknown.condition_found is False
    assert unknown.overlap_score == 0.0


def test_knowledge_graph_schema_and_end_to_end_pipeline(
    clinical_graph: nx.DiGraph,
    tmp_path: Path,
) -> None:
    node_types: set[str] = {
        str(attrs["node_type"]) for _, attrs in clinical_graph.nodes(data=True)
    }
    edge_relations: set[str] = {
        str(attrs["relation"]) for _, _, attrs in clinical_graph.edges(data=True)
    }

    assert node_types == {node_type.value for node_type in NodeType}
    assert edge_relations == {relation.value for relation in EdgeRelation}
    assert clinical_graph.nodes["I63.9"]["snomed_ct"] == "422504002"
    assert clinical_graph.nodes["lab_code:59408-5"]["loinc_code"] == "59408-5"

    report: PipelineReport = run_pipeline(results_dir=tmp_path)

    assert (tmp_path / "clinical_kg.html").exists()
    assert (tmp_path / "validation_results.json").exists()
    assert (tmp_path / "validation_report.md").exists()
    assert report.summary.total_cases == 6
    assert report.summary.all_notes_hipaa_compliant is True
    assert report.summary.total_phi_redacted == 49
    assert report.summary.validated_count == 2
    assert report.summary.contraindicated_count == 2
    assert report.summary.unverified_count == 2
