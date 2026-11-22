"""Clinical Knowledge Graph validator using Pydantic v2 models."""

from __future__ import annotations

import re
from enum import StrEnum

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from src.kg_builder import (
    EdgeRelation,
    NodeType,
    build_knowledge_graph,
    normalize_term,
)


class ValidationStatus(StrEnum):
    VALIDATED = "VALIDATED"
    CRITICAL_CONTRAINDICATION = "CRITICAL_CONTRAINDICATION"
    UNVERIFIED_INTERVENTION = "UNVERIFIED_INTERVENTION"
    UNKNOWN_CONDITION = "UNKNOWN_CONDITION"


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    icd10_code: str
    disease_name: str | None = None
    proposed_intervention: str
    status: ValidationStatus
    is_safe: bool
    matched_rule: str | None = None
    reason: str
    requires_clinician_review: bool = False


class SymptomClusterResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    icd10_code: str
    disease_name: str | None = None
    condition_found: bool
    observed_symptoms: list[str] = Field(default_factory=list)
    expected_symptoms: list[str] = Field(default_factory=list)
    matched_symptoms: list[str] = Field(default_factory=list)
    unmatched_observed_symptoms: list[str] = Field(default_factory=list)
    missing_expected_symptoms: list[str] = Field(default_factory=list)
    overlap_score: float = Field(..., ge=0.0, le=100.0)
    observed_match_pct: float = Field(..., ge=0.0, le=100.0)


# Generic single-word modifiers excluded from partial phrase containment so broad
# terms like "oral", "therapy", or "acute" cannot trigger false matches.
_GENERIC_CLINICAL_MODIFIERS: frozenset[str] = frozenset(
    {
        "acute",
        "after",
        "aggressive",
        "chronic",
        "control",
        "controlled",
        "delayed",
        "during",
        "early",
        "evaluation",
        "excessive",
        "fluids",
        "high",
        "intravenous",
        "management",
        "mild",
        "moderate",
        "monitoring",
        "oral",
        "phase",
        "prior",
        "prolonged",
        "routine",
        "severe",
        "supplemental",
        "systemic",
        "target",
        "therapy",
        "training",
        "uncontrolled",
        "unmonitored",
        "unprotected",
        "urgent",
        "using",
        "when",
        "within",
        "without",
    }
)

# Clauses introduced by these qualifiers specify conditions or timing rather than
# the primary clinical action (e.g. "oral intake prior to dysphagia screening").
_QUALIFIER_SPLIT_RE: re.Pattern[str] = re.compile(
    r"\b(?:prior to|without|during|after|if|when|preventing|triggering|impairing|due to|outside|beyond)\b"
)

# Prefixes that invert clinical polarity (e.g. "delayed non-invasive ventilation"
# contraindicates delay, not the ventilation itself).
_POLARITY_REVERSING_PREFIXES: tuple[str, ...] = ("delayed ", "premature ")


def _is_specific_phrase(normalized_phrase: str) -> bool:
    tokens: list[str] = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized_phrase)
    if not tokens:
        return False
    if len(tokens) >= 2:
        return any(tok not in _GENERIC_CLINICAL_MODIFIERS for tok in tokens)
    return len(tokens[0]) >= 4 and tokens[0] not in _GENERIC_CLINICAL_MODIFIERS


def _contains_whole_phrase(haystack: str, needle: str) -> bool:
    pattern: str = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _primary_clause(normalized_label: str) -> str:
    return _QUALIFIER_SPLIT_RE.split(normalized_label, maxsplit=1)[0].strip()


def _matches_clinical_phrase(
    query_norm: str,
    target_norm: str,
    *,
    check_polarity_prefix: bool,
) -> bool:
    if not query_norm or not target_norm:
        return False

    if _is_specific_phrase(target_norm) and _contains_whole_phrase(
        query_norm, target_norm
    ):
        return True

    if not _is_specific_phrase(query_norm):
        return False

    primary: str = _primary_clause(target_norm)
    if not _contains_whole_phrase(primary, query_norm):
        return False

    if check_polarity_prefix:
        for prefix in _POLARITY_REVERSING_PREFIXES:
            if primary.startswith(prefix) and not query_norm.startswith(prefix):
                return False

    return True


class KGValidator:
    """Validates clinical recommendations and symptom clusters against a KG."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.graph: nx.DiGraph = (
            graph if graph is not None else build_knowledge_graph()
        )

    def _resolve_disease_node(self, icd10_code: str) -> str | None:
        normalized_code: str = icd10_code.strip().upper()
        if normalized_code in self.graph:
            attrs: dict[str, object] = self.graph.nodes[normalized_code]
            if attrs.get("node_type") == NodeType.DISEASE.value:
                return normalized_code

        for node_id, attrs in self.graph.nodes(data=True):
            if (
                attrs.get("node_type") == NodeType.DISEASE.value
                and str(attrs.get("icd10_code", "")).strip().upper() == normalized_code
            ):
                return str(node_id)

        return None

    def _get_related_labels(
        self,
        disease_node: str,
        relation: EdgeRelation,
    ) -> list[str]:
        labels: list[str] = []
        for _, target, edge_attrs in self.graph.out_edges(disease_node, data=True):
            if edge_attrs.get("relation") == relation.value:
                node_attrs: dict[str, object] = self.graph.nodes[target]
                label: str = str(node_attrs.get("name") or node_attrs.get("label") or target)
                labels.append(label)
        return labels

    def validate_recommendation(
        self,
        icd10_code: str,
        proposed_intervention: str,
    ) -> ValidationResult:
        normalized_code: str = icd10_code.strip().upper()
        disease_node: str | None = self._resolve_disease_node(normalized_code)

        if disease_node is None:
            return ValidationResult(
                icd10_code=normalized_code,
                disease_name=None,
                proposed_intervention=proposed_intervention,
                status=ValidationStatus.UNKNOWN_CONDITION,
                is_safe=False,
                matched_rule=None,
                reason=(
                    f"ICD-10 code '{normalized_code}' was not found in the clinical "
                    "knowledge graph; human clinician review required."
                ),
                requires_clinician_review=True,
            )

        disease_name: str = str(
            self.graph.nodes[disease_node].get("name", disease_node)
        )
        query_norm: str = normalize_term(proposed_intervention)

        if not query_norm:
            return ValidationResult(
                icd10_code=normalized_code,
                disease_name=disease_name,
                proposed_intervention=proposed_intervention,
                status=ValidationStatus.UNVERIFIED_INTERVENTION,
                is_safe=False,
                matched_rule=None,
                reason="Empty intervention proposal; human clinician review required.",
                requires_clinician_review=True,
            )

        contraindications: list[str] = self._get_related_labels(
            disease_node, EdgeRelation.CONTRAINDICATED_WITH
        )
        interventions: list[str] = self._get_related_labels(
            disease_node, EdgeRelation.TREATED_BY
        )

        # Evaluate exact normalized matches before phrase containment, checking
        # contraindications first to maintain a safety-first decision boundary.
        for label in contraindications:
            if query_norm == normalize_term(label):
                return ValidationResult(
                    icd10_code=normalized_code,
                    disease_name=disease_name,
                    proposed_intervention=proposed_intervention,
                    status=ValidationStatus.CRITICAL_CONTRAINDICATION,
                    is_safe=False,
                    matched_rule=label,
                    reason=(
                        f"Proposed intervention matches contraindication '{label}' "
                        f"for {disease_name} ({normalized_code})."
                    ),
                    requires_clinician_review=True,
                )

        for label in interventions:
            if query_norm == normalize_term(label):
                return ValidationResult(
                    icd10_code=normalized_code,
                    disease_name=disease_name,
                    proposed_intervention=proposed_intervention,
                    status=ValidationStatus.VALIDATED,
                    is_safe=True,
                    matched_rule=label,
                    reason=(
                        f"Proposed intervention matches recommended treatment '{label}' "
                        f"for {disease_name} ({normalized_code})."
                    ),
                    requires_clinician_review=False,
                )

        for label in contraindications:
            if _matches_clinical_phrase(
                query_norm,
                normalize_term(label),
                check_polarity_prefix=True,
            ):
                return ValidationResult(
                    icd10_code=normalized_code,
                    disease_name=disease_name,
                    proposed_intervention=proposed_intervention,
                    status=ValidationStatus.CRITICAL_CONTRAINDICATION,
                    is_safe=False,
                    matched_rule=label,
                    reason=(
                        f"Proposed intervention conflicts with contraindication '{label}' "
                        f"for {disease_name} ({normalized_code})."
                    ),
                    requires_clinician_review=True,
                )

        for label in interventions:
            if _matches_clinical_phrase(
                query_norm,
                normalize_term(label),
                check_polarity_prefix=False,
            ):
                return ValidationResult(
                    icd10_code=normalized_code,
                    disease_name=disease_name,
                    proposed_intervention=proposed_intervention,
                    status=ValidationStatus.VALIDATED,
                    is_safe=True,
                    matched_rule=label,
                    reason=(
                        f"Proposed intervention aligns with recommended treatment '{label}' "
                        f"for {disease_name} ({normalized_code})."
                    ),
                    requires_clinician_review=False,
                )

        return ValidationResult(
            icd10_code=normalized_code,
            disease_name=disease_name,
            proposed_intervention=proposed_intervention,
            status=ValidationStatus.UNVERIFIED_INTERVENTION,
            is_safe=False,
            matched_rule=None,
            reason=(
                f"Intervention '{proposed_intervention}' is not established in the KG "
                f"for {disease_name} ({normalized_code}); flagged for human clinician review."
            ),
            requires_clinician_review=True,
        )

    def validate_symptom_cluster(
        self,
        icd10_code: str,
        observed_symptoms: list[str],
    ) -> SymptomClusterResult:
        normalized_code: str = icd10_code.strip().upper()
        disease_node: str | None = self._resolve_disease_node(normalized_code)

        cleaned_observed: list[str] = [
            symptom.strip() for symptom in observed_symptoms if symptom.strip()
        ]

        if disease_node is None:
            return SymptomClusterResult(
                icd10_code=normalized_code,
                disease_name=None,
                condition_found=False,
                observed_symptoms=cleaned_observed,
                expected_symptoms=[],
                matched_symptoms=[],
                unmatched_observed_symptoms=cleaned_observed,
                missing_expected_symptoms=[],
                overlap_score=0.0,
                observed_match_pct=0.0,
            )

        disease_name: str = str(
            self.graph.nodes[disease_node].get("name", disease_node)
        )
        expected_symptoms: list[str] = self._get_related_labels(
            disease_node, EdgeRelation.HAS_SYMPTOM
        )
        expected_norm_pairs: list[tuple[str, str]] = [
            (label, normalize_term(label)) for label in expected_symptoms
        ]

        matched_expected_set: set[str] = set()
        unmatched_observed: list[str] = []
        matched_observed_count: int = 0

        for observed in cleaned_observed:
            obs_norm: str = normalize_term(observed)
            matched_label: str | None = None

            for exp_label, exp_norm in expected_norm_pairs:
                if obs_norm == exp_norm:
                    matched_label = exp_label
                    break

            if matched_label is None:
                for exp_label, exp_norm in expected_norm_pairs:
                    if (
                        _is_specific_phrase(obs_norm)
                        and _contains_whole_phrase(exp_norm, obs_norm)
                    ) or (
                        _is_specific_phrase(exp_norm)
                        and _contains_whole_phrase(obs_norm, exp_norm)
                    ):
                        matched_label = exp_label
                        break

            if matched_label is not None:
                matched_expected_set.add(matched_label)
                matched_observed_count += 1
            else:
                unmatched_observed.append(observed)

        matched_symptoms: list[str] = [
            label for label in expected_symptoms if label in matched_expected_set
        ]
        missing_expected: list[str] = [
            label for label in expected_symptoms if label not in matched_expected_set
        ]

        overlap_score: float = (
            round((len(matched_symptoms) / len(expected_symptoms)) * 100.0, 2)
            if expected_symptoms
            else 0.0
        )
        observed_match_pct: float = (
            round((matched_observed_count / len(cleaned_observed)) * 100.0, 2)
            if cleaned_observed
            else 0.0
        )

        return SymptomClusterResult(
            icd10_code=normalized_code,
            disease_name=disease_name,
            condition_found=True,
            observed_symptoms=cleaned_observed,
            expected_symptoms=expected_symptoms,
            matched_symptoms=matched_symptoms,
            unmatched_observed_symptoms=unmatched_observed,
            missing_expected_symptoms=missing_expected,
            overlap_score=overlap_score,
            observed_match_pct=observed_match_pct,
        )
