"""Knowledge Graph builder and interactive exporter for clinical ontologies."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_ONTOLOGY_PATH: Path = Path("ontologies/clinical_ontologies.json")
DEFAULT_HTML_OUTPUT_PATH: Path = Path("results/clinical_kg.html")


class NodeType(StrEnum):
    DISEASE = "disease"
    SYMPTOM = "symptom"
    INTERVENTION = "intervention"
    CONTRAINDICATION = "contraindication"
    LAB_CODE = "lab_code"


class EdgeRelation(StrEnum):
    HAS_SYMPTOM = "HAS_SYMPTOM"
    TREATED_BY = "TREATED_BY"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"


NODE_TYPE_COLORS: dict[str, str] = {
    NodeType.DISEASE.value: "#1d4ed8",
    NodeType.SYMPTOM.value: "#d97706",
    NodeType.INTERVENTION.value: "#059669",
    NodeType.CONTRAINDICATION.value: "#dc2626",
    NodeType.LAB_CODE.value: "#7c3aed",
}

NODE_TYPE_SIZES: dict[str, int] = {
    NodeType.DISEASE.value: 28,
    NodeType.SYMPTOM.value: 16,
    NodeType.INTERVENTION.value: 18,
    NodeType.CONTRAINDICATION.value: 18,
    NodeType.LAB_CODE.value: 18,
}

EDGE_RELATION_COLORS: dict[str, str] = {
    EdgeRelation.HAS_SYMPTOM.value: "#d97706",
    EdgeRelation.TREATED_BY.value: "#059669",
    EdgeRelation.CONTRAINDICATED_WITH.value: "#dc2626",
}


class ReferenceRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    min: float
    max: float


class DiseaseEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    icd10_code: str = Field(..., min_length=1)
    snomed_ct: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    clinical_domain: str = "general"
    expected_symptoms: list[str] = Field(default_factory=list)
    recommended_interventions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)


class LoincLabEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    loinc_code: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    short_name: str | None = None
    unit: str = ""
    reference_range: ReferenceRange | None = None
    critical_low: float | None = None
    critical_high: float | None = None
    clinical_significance: str = ""


class ClinicalOntology(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: dict[str, Any] = Field(default_factory=dict)
    diseases: list[DiseaseEntry] = Field(default_factory=list)
    loinc_labs: list[LoincLabEntry] = Field(default_factory=list)


_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


def normalize_term(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def load_ontology(ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH) -> ClinicalOntology:
    path: Path = Path(ontology_path)
    raw_data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return ClinicalOntology.model_validate(raw_data)


def build_graph_from_ontology(ontology: ClinicalOntology) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph(
        ontology_version=ontology.metadata.get("version", "unknown"),
        domains=ontology.metadata.get("domains", []),
    )

    for disease in ontology.diseases:
        disease_id: str = disease.icd10_code.strip().upper()
        graph.add_node(
            disease_id,
            node_type=NodeType.DISEASE.value,
            label=disease.name,
            name=disease.name,
            icd10_code=disease_id,
            snomed_ct=disease.snomed_ct.strip(),
            loinc_code=None,
            clinical_domain=disease.clinical_domain,
        )

        # Namespace concept node IDs by type so an intervention for one condition
        # never overwrites node_type if the same phrase is contraindicated elsewhere.
        for symptom in disease.expected_symptoms:
            symptom_id: str = f"{NodeType.SYMPTOM.value}:{normalize_term(symptom)}"
            if symptom_id not in graph:
                graph.add_node(
                    symptom_id,
                    node_type=NodeType.SYMPTOM.value,
                    label=symptom.strip(),
                    name=symptom.strip(),
                    icd10_code=None,
                    snomed_ct=None,
                    loinc_code=None,
                )
            graph.add_edge(
                disease_id,
                symptom_id,
                relation=EdgeRelation.HAS_SYMPTOM.value,
                label=EdgeRelation.HAS_SYMPTOM.value,
            )

        for intervention in disease.recommended_interventions:
            intervention_id: str = (
                f"{NodeType.INTERVENTION.value}:{normalize_term(intervention)}"
            )
            if intervention_id not in graph:
                graph.add_node(
                    intervention_id,
                    node_type=NodeType.INTERVENTION.value,
                    label=intervention.strip(),
                    name=intervention.strip(),
                    icd10_code=None,
                    snomed_ct=None,
                    loinc_code=None,
                )
            graph.add_edge(
                disease_id,
                intervention_id,
                relation=EdgeRelation.TREATED_BY.value,
                label=EdgeRelation.TREATED_BY.value,
            )

        for contraindication in disease.contraindications:
            contra_id: str = (
                f"{NodeType.CONTRAINDICATION.value}:{normalize_term(contraindication)}"
            )
            if contra_id not in graph:
                graph.add_node(
                    contra_id,
                    node_type=NodeType.CONTRAINDICATION.value,
                    label=contraindication.strip(),
                    name=contraindication.strip(),
                    icd10_code=None,
                    snomed_ct=None,
                    loinc_code=None,
                )
            graph.add_edge(
                disease_id,
                contra_id,
                relation=EdgeRelation.CONTRAINDICATED_WITH.value,
                label=EdgeRelation.CONTRAINDICATED_WITH.value,
            )

    for lab in ontology.loinc_labs:
        loinc_code: str = lab.loinc_code.strip()
        lab_id: str = f"{NodeType.LAB_CODE.value}:{loinc_code}"
        display_name: str = lab.short_name or lab.name
        graph.add_node(
            lab_id,
            node_type=NodeType.LAB_CODE.value,
            label=f"{display_name} ({loinc_code})",
            name=lab.name,
            short_name=display_name,
            icd10_code=None,
            snomed_ct=None,
            loinc_code=loinc_code,
            unit=lab.unit,
            reference_range=(
                lab.reference_range.model_dump() if lab.reference_range else None
            ),
            critical_low=lab.critical_low,
            critical_high=lab.critical_high,
            clinical_significance=lab.clinical_significance,
        )

    return graph


def build_knowledge_graph(
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> nx.DiGraph:
    ontology: ClinicalOntology = load_ontology(ontology_path)
    return build_graph_from_ontology(ontology)


def _build_node_tooltip(node_id: str, attrs: dict[str, Any]) -> str:
    node_type: str = str(attrs.get("node_type", "unknown"))
    lines: list[str] = [
        f"Type: {node_type}",
        f"Name: {attrs.get('name', node_id)}",
    ]
    if attrs.get("icd10_code"):
        lines.append(f"ICD-10: {attrs['icd10_code']}")
    if attrs.get("snomed_ct"):
        lines.append(f"SNOMED CT: {attrs['snomed_ct']}")
    if attrs.get("loinc_code"):
        lines.append(f"LOINC: {attrs['loinc_code']}")
    if attrs.get("unit"):
        lines.append(f"Unit: {attrs['unit']}")
    ref_range: dict[str, float] | None = attrs.get("reference_range")
    if ref_range:
        lines.append(f"Reference Range: {ref_range['min']} - {ref_range['max']}")
    if attrs.get("clinical_significance"):
        lines.append(f"Clinical Note: {attrs['clinical_significance']}")
    return "\n".join(lines)


def export_interactive_graph(
    graph: nx.DiGraph,
    output_path: str | Path = DEFAULT_HTML_OUTPUT_PATH,
) -> Path:
    from pyvis.network import Network

    net: Network = Network(
        height="800px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f8fafc",
        font_color="#0f172a",
        cdn_resources="remote",
    )
    net.toggle_physics(True)
    net.barnes_hut(
        gravity=-12000,
        central_gravity=0.35,
        spring_length=180,
        spring_strength=0.02,
        damping=0.12,
        overlap=0.1,
    )

    for node_id, attrs in graph.nodes(data=True):
        node_type: str = str(attrs.get("node_type", "unknown"))
        color: str = NODE_TYPE_COLORS.get(node_type, "#64748b")
        size: int = NODE_TYPE_SIZES.get(node_type, 16)
        shape: str = "diamond" if node_type == NodeType.LAB_CODE.value else "dot"
        label: str = str(attrs.get("label", node_id))
        title: str = _build_node_tooltip(str(node_id), attrs)

        net.add_node(
            str(node_id),
            label=label,
            title=title,
            color=color,
            size=size,
            shape=shape,
            group=node_type,
        )

    for source, target, edge_attrs in graph.edges(data=True):
        relation: str = str(edge_attrs.get("relation", ""))
        edge_color: str = EDGE_RELATION_COLORS.get(relation, "#94a3b8")
        dashes: bool = relation == EdgeRelation.CONTRAINDICATED_WITH.value
        net.add_edge(
            str(source),
            str(target),
            label=relation,
            title=relation,
            color=edge_color,
            arrows="to",
            dashes=dashes,
        )

    destination: Path = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html_content: str = net.generate_html()
    destination.write_text(html_content, encoding="utf-8")
    return destination
