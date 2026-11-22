"""End-to-end execution pipeline for med-ontology-kg-validator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from src.hipaa_deidentifier import (
    DeidentificationResult,
    HIPAADeidentifier,
    PHICategory,
)
from src.kg_builder import (
    DEFAULT_ONTOLOGY_PATH,
    build_knowledge_graph,
    export_interactive_graph,
)
from src.validator import (
    KGValidator,
    SymptomClusterResult,
    ValidationResult,
    ValidationStatus,
)


DEFAULT_CASES_PATH: Path = Path("data/sample_clinical_cases.json")
DEFAULT_RESULTS_DIR: Path = Path("results")

logger: logging.Logger = logging.getLogger(__name__)


class ClinicalCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., min_length=1)
    raw_clinical_note: str
    icd10_code: str = Field(..., min_length=1)
    observed_symptoms: list[str] = Field(default_factory=list)
    ai_proposed_intervention: str


class CaseValidationOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    icd10_code: str
    disease_name: str | None
    deidentification: DeidentificationResult
    recommendation_validation: ValidationResult
    symptom_validation: SymptomClusterResult


class PipelineSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(..., ge=0)
    graph_nodes: int = Field(..., ge=0)
    graph_edges: int = Field(..., ge=0)
    total_phi_redacted: int = Field(..., ge=0)
    phi_by_category: dict[str, int] = Field(default_factory=dict)
    all_notes_hipaa_compliant: bool
    validated_count: int = Field(..., ge=0)
    contraindicated_count: int = Field(..., ge=0)
    unverified_count: int = Field(..., ge=0)
    unknown_condition_count: int = Field(..., ge=0)
    average_symptom_overlap_pct: float = Field(..., ge=0.0, le=100.0)


class PipelineReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: PipelineSummary
    cases: list[CaseValidationOutput] = Field(default_factory=list)


def load_clinical_cases(cases_path: Path = DEFAULT_CASES_PATH) -> list[ClinicalCase]:
    raw_payload: Any = json.loads(cases_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = (
        raw_payload["cases"]
        if isinstance(raw_payload, dict) and "cases" in raw_payload
        else raw_payload
    )
    return [ClinicalCase.model_validate(item) for item in records]


def evaluate_cases(
    cases: list[ClinicalCase],
    deidentifier: HIPAADeidentifier,
    validator: KGValidator,
) -> list[CaseValidationOutput]:
    outputs: list[CaseValidationOutput] = []

    for case in cases:
        deid_result: DeidentificationResult = deidentifier.deidentify(
            case.raw_clinical_note
        )
        rec_result: ValidationResult = validator.validate_recommendation(
            icd10_code=case.icd10_code,
            proposed_intervention=case.ai_proposed_intervention,
        )
        symptom_result: SymptomClusterResult = validator.validate_symptom_cluster(
            icd10_code=case.icd10_code,
            observed_symptoms=case.observed_symptoms,
        )
        outputs.append(
            CaseValidationOutput(
                case_id=case.case_id,
                icd10_code=rec_result.icd10_code,
                disease_name=rec_result.disease_name,
                deidentification=deid_result,
                recommendation_validation=rec_result,
                symptom_validation=symptom_result,
            )
        )

    return outputs


def build_summary(
    graph: nx.DiGraph,
    outputs: list[CaseValidationOutput],
) -> PipelineSummary:
    phi_by_category: dict[str, int] = {cat.value: 0 for cat in PHICategory}
    for item in outputs:
        for category, count in item.deidentification.entities_redacted.items():
            phi_by_category[category] = phi_by_category.get(category, 0) + count

    total_phi_redacted: int = sum(phi_by_category.values())
    all_compliant: bool = all(
        item.deidentification.is_hipaa_compliant for item in outputs
    )

    validated_count: int = sum(
        1
        for item in outputs
        if item.recommendation_validation.status == ValidationStatus.VALIDATED
    )
    contraindicated_count: int = sum(
        1
        for item in outputs
        if item.recommendation_validation.status
        == ValidationStatus.CRITICAL_CONTRAINDICATION
    )
    unverified_count: int = sum(
        1
        for item in outputs
        if item.recommendation_validation.status
        == ValidationStatus.UNVERIFIED_INTERVENTION
    )
    unknown_count: int = sum(
        1
        for item in outputs
        if item.recommendation_validation.status == ValidationStatus.UNKNOWN_CONDITION
    )

    avg_overlap: float = (
        round(
            sum(item.symptom_validation.overlap_score for item in outputs)
            / len(outputs),
            2,
        )
        if outputs
        else 0.0
    )

    return PipelineSummary(
        total_cases=len(outputs),
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        total_phi_redacted=total_phi_redacted,
        phi_by_category=phi_by_category,
        all_notes_hipaa_compliant=all_compliant,
        validated_count=validated_count,
        contraindicated_count=contraindicated_count,
        unverified_count=unverified_count,
        unknown_condition_count=unknown_count,
        average_symptom_overlap_pct=avg_overlap,
    )


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def render_markdown_report(report: PipelineReport) -> str:
    s: PipelineSummary = report.summary
    lines: list[str] = [
        "# Clinical Knowledge Graph Validation Report",
        "",
        "## Pipeline Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Knowledge Graph Nodes | {s.graph_nodes} |",
        f"| Knowledge Graph Edges | {s.graph_edges} |",
        f"| Clinical Cases Evaluated | {s.total_cases} |",
        f"| Total PHI Entities Redacted | {s.total_phi_redacted} |",
        f"| All Notes HIPAA Compliant | {'Yes' if s.all_notes_hipaa_compliant else 'No'} |",
        f"| Validated Interventions | {s.validated_count} |",
        f"| Critical Contraindications | {s.contraindicated_count} |",
        f"| Unverified Interventions (Clinician Review) | {s.unverified_count} |",
        f"| Unknown Conditions | {s.unknown_condition_count} |",
        f"| Mean Symptom Cluster Overlap (%) | {s.average_symptom_overlap_pct:.2f}% |",
        "",
        "## Case Validation Results",
        "",
        (
            "| Case ID | ICD-10 | Disease | AI Proposed Intervention | Status | "
            "Safe | Symptom Overlap (%) | PHI Redacted | HIPAA Compliant |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]

    for item in report.cases:
        rec: ValidationResult = item.recommendation_validation
        sym: SymptomClusterResult = item.symptom_validation
        phi_count: int = sum(item.deidentification.entities_redacted.values())
        disease_label: str = item.disease_name or "Unknown"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md_cell(item.case_id),
                    _escape_md_cell(item.icd10_code),
                    _escape_md_cell(disease_label),
                    _escape_md_cell(rec.proposed_intervention),
                    _escape_md_cell(rec.status.value),
                    "Yes" if rec.is_safe else "No",
                    f"{sym.overlap_score:.2f}%",
                    str(phi_count),
                    "Yes" if item.deidentification.is_hipaa_compliant else "No",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Clinical Safety Rationales", ""])
    for item in report.cases:
        rec = item.recommendation_validation
        matched: str = rec.matched_rule if rec.matched_rule else "None"
        lines.append(
            f"- **{item.case_id} ({item.icd10_code})** [{rec.status.value}]: "
            f"{rec.reason} (Matched KG rule: `{matched}`)"
        )

    lines.append("")
    return "\n".join(lines)


def write_json_results(report: PipelineReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_markdown_report(report: PipelineReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(report), encoding="utf-8")
    return output_path


def print_execution_summary(
    summary: PipelineSummary,
    html_path: Path,
    json_path: Path,
    md_path: Path,
) -> None:
    print("=== med-ontology-kg-validator Execution Summary ===")
    print(f"Knowledge Graph      : {summary.graph_nodes} nodes, {summary.graph_edges} edges")
    print(f"Cases Evaluated      : {summary.total_cases}")
    print(
        f"PHI Entities Redacted: {summary.total_phi_redacted} "
        f"(HIPAA compliant: {summary.all_notes_hipaa_compliant})"
    )
    print(f"  PHI Breakdown      : {summary.phi_by_category}")
    print(f"Validated            : {summary.validated_count}")
    print(f"Contraindicated      : {summary.contraindicated_count}")
    print(f"Unverified (Review)  : {summary.unverified_count}")
    if summary.unknown_condition_count:
        print(f"Unknown Conditions   : {summary.unknown_condition_count}")
    print(f"Mean Symptom Overlap : {summary.average_symptom_overlap_pct:.2f}%")
    print(f"Interactive KG HTML  : {html_path}")
    print(f"Structured JSON      : {json_path}")
    print(f"Markdown Report      : {md_path}")


def run_pipeline(
    cases_path: Path = DEFAULT_CASES_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> PipelineReport:
    results_dir.mkdir(parents=True, exist_ok=True)

    cases: list[ClinicalCase] = load_clinical_cases(cases_path)
    logger.info("Loaded %d synthetic clinical cases from %s", len(cases), cases_path)

    graph: nx.DiGraph = build_knowledge_graph(ontology_path)
    html_path: Path = export_interactive_graph(
        graph, output_path=results_dir / "clinical_kg.html"
    )
    logger.info(
        "Built KG (%d nodes, %d edges) and exported interactive graph to %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        html_path,
    )

    deidentifier: HIPAADeidentifier = HIPAADeidentifier()
    validator: KGValidator = KGValidator(graph)
    outputs: list[CaseValidationOutput] = evaluate_cases(cases, deidentifier, validator)

    summary: PipelineSummary = build_summary(graph, outputs)
    logger.info(
        "De-identified %d clinical notes; redacted %d total PHI entities (HIPAA compliant=%s)",
        summary.total_cases,
        summary.total_phi_redacted,
        summary.all_notes_hipaa_compliant,
    )

    report: PipelineReport = PipelineReport(summary=summary, cases=outputs)
    json_path: Path = write_json_results(
        report, results_dir / "validation_results.json"
    )
    md_path: Path = write_markdown_report(
        report, results_dir / "validation_report.md"
    )

    print_execution_summary(summary, html_path, json_path, md_path)
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
