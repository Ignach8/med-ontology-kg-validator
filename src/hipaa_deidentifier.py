"""HIPAA Safe Harbor de-identification pipeline for clinical text."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PHICategory(StrEnum):
    NAME = "name"
    DATE = "date"
    NATIONAL_ID = "national_id"
    PHONE = "phone"
    EMAIL = "email"
    MRN = "mrn"


class DeidentifierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    redact_names: bool = True
    redact_dates: bool = True
    redact_national_ids: bool = True
    redact_phones: bool = True
    redact_emails: bool = True
    redact_mrns: bool = True
    propagate_full_names: bool = True


class DeidentificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    original_length: int = Field(..., ge=0)
    redacted_text: str
    entities_redacted: dict[str, int] = Field(default_factory=dict)
    is_hipaa_compliant: bool


_UPPER: str = r"A-ZÁÉÍÓÚÜÑ"
_LOWER: str = r"a-záéíóúüñ"

_MONTHS: str = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_NAME_PARTICLES: str = r"(?:de[ \t]+la|de[ \t]+los|de[ \t]+las|de|del|da|dos|van|von)"

# Trailing negative lookahead prevents partial matches on mixed-alphanumeric clinical
# tokens such as SpO2, PaCO2, or ICD-10 codes.
_TITLE_NAME_TOKEN: str = (
    rf"(?:[{_UPPER}][{_LOWER}]+(?:['-][{_UPPER}]?[{_LOWER}]+|[{_UPPER}][{_LOWER}]+)*"
    rf"|[{_UPPER}]'[{_UPPER}][{_LOWER}]+"
    rf"|[{_UPPER}]\.)"
    rf"(?![A-Za-z0-9{_UPPER}{_LOWER}])"
)

_ALLCAPS_NAME_TOKEN: str = (
    rf"[{_UPPER}]{{2,}}(?:['-][{_UPPER}]{{2,}})*(?![A-Za-z0-9{_UPPER}{_LOWER}])"
)

_NAME_TOKEN: str = rf"(?:{_TITLE_NAME_TOKEN}|{_ALLCAPS_NAME_TOKEN})"

_FULL_NAME_SEQUENCE: str = (
    rf"{_NAME_TOKEN}(?:[ \t]+(?:{_NAME_PARTICLES}[ \t]+)?{_NAME_TOKEN}){{0,3}}"
)

# Clinical credentials and diagnostic acronyms excluded from all-caps name matching
# so adjacent chart tokens (e.g. "Dr. Smith MD", "Patient: JUAN PEREZ CT") survive.
_CLINICAL_ACRONYMS: frozenset[str] = frozenset(
    {
        "ABG",
        "ARDS",
        "BIPAP",
        "BMP",
        "BP",
        "CAP",
        "CBC",
        "CMP",
        "COPD",
        "CT",
        "DBP",
        "DO",
        "DOB",
        "DVT",
        "ECG",
        "ED",
        "EKG",
        "ER",
        "FVC",
        "GBS",
        "GCS",
        "HR",
        "ICH",
        "ICU",
        "ID",
        "IM",
        "INR",
        "IV",
        "IVIG",
        "MAP",
        "MD",
        "MG",
        "MRI",
        "MRN",
        "NIF",
        "NIHSS",
        "NIV",
        "NP",
        "PA",
        "PBW",
        "PE",
        "PEEP",
        "PHD",
        "PO",
        "RBC",
        "RN",
        "RR",
        "RUN",
        "RUT",
        "SBP",
        "SCI",
        "SSN",
        "TBI",
        "UTI",
        "WBC",
    }
)

# Clinical narrative, demographic, and diagnostic starters that commonly appear
# after "Patient:" or a comma-separated surname without being a personal name.
_PATIENT_NON_NAME_STARTERS: frozenset[str] = frozenset(
    {
        "acute",
        "admitted",
        "age",
        "alert",
        "awake",
        "bed",
        "cervical",
        "chronic",
        "community-acquired",
        "complains",
        "copd",
        "denies",
        "developed",
        "diagnosis",
        "discharged",
        "female",
        "guillain-barre",
        "has",
        "hemorrhagic",
        "history",
        "intracerebral",
        "intubated",
        "is",
        "ischemic",
        "male",
        "mild",
        "moderate",
        "myasthenia",
        "no",
        "none",
        "pneumonia",
        "presented",
        "presents",
        "refuses",
        "reports",
        "requires",
        "room",
        "severe",
        "sex",
        "stable",
        "status",
        "stroke",
        "syndrome",
        "tolerates",
        "unit",
        "unknown",
        "unstable",
        "ward",
        "was",
    }
)

_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Require at least one digit in the MRN value so clinical prose such as
# "Medical Record Number pending" is not falsely redacted.
_MRN_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:MRN|Medical[ \t]+Record[ \t]+(?:Number|No\.?|#)|Med[ \t]+Rec[ \t]*(?:No\.?|#))"
    r"[ \t]*[:#-]?(?:[ \t]+|(?=[A-Za-z0-9]))"
    r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,19}\b",
    re.IGNORECASE,
)

# Labeled national IDs cover SSN, Chilean RUT/RUN, DNI, and passport identifiers
# while requiring at least one digit to avoid matching status words.
_LABELED_NATIONAL_ID_PATTERN: re.Pattern[str] = re.compile(
    r"(\b(?:SSN|Social[ \t]+Security(?:[ \t]+Number)?|RUT|RUN|DNI|"
    r"National[ \t]+ID|C[eé]dula(?:[ \t]+de[ \t]+Identidad)?|"
    r"Passport(?:[ \t]+(?:Number|No\.?))?)(?:[ \t]*[:#][ \t]*|[ \t]+))"
    r"(?=[A-Za-z0-9.-]*\d)"
    r"(?:\d{3}[- ]?\d{2}[- ]?\d{4}"
    r"|(?:\d{1,2}\.\d{3}\.\d{3}|\d{3}\.\d{3}|\d{7,8})-?[0-9Kk]"
    r"|[A-Za-z0-9][A-Za-z0-9.-]{5,17})\b",
    re.IGNORECASE,
)

# Unlabeled Chilean RUTs without dots require 7-8 digits before the check-digit
# hyphen so 4-6 digit LOINC codes (e.g. 59408-5, 2744-1) are never redacted.
_UNLABELED_NATIONAL_ID_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"\d{3}-\d{2}-\d{4}"
    r"|(?:\d{1,2}\.\d{3}\.\d{3}|\d{3}\.\d{3})-[0-9Kk]"
    r"|\d{7,8}-[0-9Kk]"
    r")\b"
)

# Require three date components to prevent false positives on clinical ratios
# (e.g. blood pressure 120/80, motor grade 4/5) and preserve standalone years
# permitted under HIPAA Safe Harbor (45 CFR 164.514(b)(2)).
_DATE_PATTERN: re.Pattern[str] = re.compile(
    rf"(?:"
    rf"\b(?:19|20)\d{{2}}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"
    rf"|\b(?:(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])|(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2]))[-/](?:19|20)\d{{2}}\b"
    rf"|\b(?:(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])|(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2]))\.(?:19|20)\d{{2}}\b"
    rf"|\b(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])|(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2]))/\d{{2}}\b"
    rf"|\b{_MONTHS}\.?[ \t]+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?[ \t]+(?:19|20)\d{{2}}\b"
    rf"|\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[ \t]+(?:of[ \t]+)?{_MONTHS}\.?,?[ \t]+(?:19|20)\d{{2}}\b"
    rf")",
    re.IGNORECASE,
)

_LABELED_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(\b(?:Phone|Tel(?:ephone)?|Cell|Mobile|Fax|Fono|Cel)\.?(?:[ \t]*[:#][ \t]*|[ \t]+))"
    r"(\+?\d[\d \t().-]{6,20}\d)\b",
    re.IGNORECASE,
)

# Require an international '+' prefix or standard telephone grouping separators so
# bare 9-digit SNOMED CT codes (e.g. 422504002) are never misclassified as phones.
_UNLABELED_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\w)(?:"
    r"\+\d{1,3}(?:[ \t.-]?\(?\d{1,4}\)?){2,5}"
    r"|(?:1[ \t.-])?(?:\(\d{3}\)[ \t.-]?\d{3}[ \t.-]\d{4}|\d{3}[ \t.-]\d{3}[ \t.-]\d{4})"
    r"|9[ \t.-]\d{4}[ \t.-]\d{4}"
    r")\b"
)

# Match honorifics in title case or all-caps with a mandatory period so bare
# clinical acronyms like "MR Angiography" (Magnetic Resonance) are not matched.
_HONORIFIC_NAME_PATTERN: re.Pattern[str] = re.compile(
    rf"(\b(?:(?:Mr|Mrs|Ms|Miss|Dr|Dra|Sr|Sra|Prof)\.?|(?:MR|MRS|MS|MISS|DR|DRA|SR|SRA|PROF)\.)[ \t]+)"
    rf"({_FULL_NAME_SEQUENCE})"
)

# Support both "First Last" and "Last, First" after explicit patient labels while
# blocking "Last, Field:" section headers via negative lookahead for a colon.
_LABELED_PATIENT_NAME_PATTERN: re.Pattern[str] = re.compile(
    rf"((?i:\b(?:Patient(?:[ \t]+Name)?|Full[ \t]+Name|Pt\.?|Paciente|Subject))[ \t]*:[ \t]*)"
    rf"({_NAME_TOKEN},[ \t]+{_NAME_TOKEN}(?:[ \t]+{_NAME_TOKEN})?(?![ \t]*:)|{_FULL_NAME_SEQUENCE})"
)


def _count_digits(value: str) -> int:
    return sum(ch.isdigit() for ch in value)


def _trim_clinical_suffix(candidate: str) -> tuple[str, str]:
    trailing_suffix: str = ""
    working: str = candidate

    if "," in working:
        before_comma, after_comma = working.split(",", 1)
        post_tokens: list[str] = after_comma.strip().split()
        if post_tokens and (
            post_tokens[0].lower() in _PATIENT_NON_NAME_STARTERS
            or (post_tokens[0].isupper() and post_tokens[0] in _CLINICAL_ACRONYMS)
        ):
            working = before_comma
            trailing_suffix = f",{after_comma}"

    parts: list[str] = re.split(r"([ \t]+)", working)
    suffix_parts: list[str] = []

    while len(parts) >= 3:
        last_token: str = parts[-1]
        if last_token.isupper() and last_token in _CLINICAL_ACRONYMS:
            suffix_parts.insert(0, parts.pop())
            suffix_parts.insert(0, parts.pop())
        else:
            break

    return "".join(parts), "".join(suffix_parts) + trailing_suffix


def _is_valid_name_candidate(name: str, *, allow_stopword_single: bool) -> bool:
    tokens: list[str] = [tok.strip(",") for tok in name.split() if tok.strip(",")]
    if not tokens:
        return False
    if all(tok.isupper() and tok in _CLINICAL_ACRONYMS for tok in tokens):
        return False
    if not allow_stopword_single and len(tokens) == 1:
        if tokens[0].lower() in _PATIENT_NON_NAME_STARTERS:
            return False
    return True


class HIPAADeidentifier:
    """Rule-based HIPAA Safe Harbor de-identification pipeline."""

    def __init__(self, config: DeidentifierConfig | None = None) -> None:
        self.config: DeidentifierConfig = config or DeidentifierConfig()

    def deidentify(self, text: str) -> DeidentificationResult:
        counts: dict[str, int] = {category.value: 0 for category in PHICategory}
        redacted: str = text

        # Apply rules from most structurally constrained to most ambiguous to prevent
        # overlapping matches (e.g. digits inside emails or MRNs matching dates/phones).
        if self.config.redact_emails:
            redacted, n = self._redact_emails(redacted)
            counts[PHICategory.EMAIL.value] += n

        if self.config.redact_mrns:
            redacted, n = self._redact_mrns(redacted)
            counts[PHICategory.MRN.value] += n

        if self.config.redact_national_ids:
            redacted, n = self._redact_national_ids(redacted)
            counts[PHICategory.NATIONAL_ID.value] += n

        if self.config.redact_dates:
            redacted, n = self._redact_dates(redacted)
            counts[PHICategory.DATE.value] += n

        if self.config.redact_phones:
            redacted, n = self._redact_phones(redacted)
            counts[PHICategory.PHONE.value] += n

        if self.config.redact_names:
            redacted, n = self._redact_names(redacted)
            counts[PHICategory.NAME.value] += n

        return DeidentificationResult(
            original_length=len(text),
            redacted_text=redacted,
            entities_redacted=counts,
            is_hipaa_compliant=self.verify_compliance(redacted),
        )

    def verify_compliance(self, text: str) -> bool:
        """Verify that no detectable Safe Harbor PHI patterns remain in text."""
        if _EMAIL_PATTERN.search(text):
            return False
        if _MRN_PATTERN.search(text):
            return False
        if _LABELED_NATIONAL_ID_PATTERN.search(text) or _UNLABELED_NATIONAL_ID_PATTERN.search(text):
            return False
        if _DATE_PATTERN.search(text):
            return False
        if self._has_phone_match(text):
            return False
        if self._has_name_match(text):
            return False
        return True

    def _redact_emails(self, text: str) -> tuple[str, int]:
        return _EMAIL_PATTERN.subn("[EMAIL]", text)

    def _redact_mrns(self, text: str) -> tuple[str, int]:
        return _MRN_PATTERN.subn("[MRN]", text)

    def _redact_national_ids(self, text: str) -> tuple[str, int]:
        text, labeled_count = _LABELED_NATIONAL_ID_PATTERN.subn(r"\1[ID_REDACTED]", text)
        text, unlabeled_count = _UNLABELED_NATIONAL_ID_PATTERN.subn("[ID_REDACTED]", text)
        return text, labeled_count + unlabeled_count

    def _redact_dates(self, text: str) -> tuple[str, int]:
        return _DATE_PATTERN.subn("[DATE]", text)

    def _redact_phones(self, text: str) -> tuple[str, int]:
        count: int = 0

        def replace_labeled(match: re.Match[str]) -> str:
            nonlocal count
            digits: int = _count_digits(match.group(2))
            # Enforce ITU-T E.164 upper bound (15 digits) and local 7-digit minimum.
            if 7 <= digits <= 15:
                count += 1
                return f"{match.group(1)}[PHONE]"
            return match.group(0)

        def replace_unlabeled(match: re.Match[str]) -> str:
            nonlocal count
            value: str = match.group(0)
            digits: int = _count_digits(value)
            min_digits: int = 8 if value.startswith("+") else 9
            if min_digits <= digits <= 15:
                count += 1
                return "[PHONE]"
            return value

        text = _LABELED_PHONE_PATTERN.sub(replace_labeled, text)
        text = _UNLABELED_PHONE_PATTERN.sub(replace_unlabeled, text)
        return text, count

    def _redact_names(self, text: str) -> tuple[str, int]:
        count: int = 0
        captured_full_names: set[str] = set()

        def replace_name(match: re.Match[str], *, allow_stopword_single: bool) -> str:
            nonlocal count
            prefix: str = match.group(1)
            candidate, trailing_suffix = _trim_clinical_suffix(match.group(2))
            if not _is_valid_name_candidate(
                candidate, allow_stopword_single=allow_stopword_single
            ):
                return match.group(0)

            count += 1
            tokens: list[str] = [tok.strip(",") for tok in candidate.split() if tok.strip(",")]
            # Only propagate multi-token full names so single surnames (e.g. "White")
            # cannot corrupt clinical findings like "White blood cell count".
            if len(tokens) >= 2 and "," not in candidate:
                captured_full_names.add(candidate)

            return f"{prefix}[NAME]{trailing_suffix}"

        text = _LABELED_PATIENT_NAME_PATTERN.sub(
            lambda m: replace_name(m, allow_stopword_single=False),
            text,
        )
        text = _HONORIFIC_NAME_PATTERN.sub(
            lambda m: replace_name(m, allow_stopword_single=True),
            text,
        )

        if self.config.propagate_full_names and captured_full_names:
            for full_name in sorted(captured_full_names, key=len, reverse=True):
                escaped: str = re.escape(full_name)
                propagated_pattern: re.Pattern[str] = re.compile(
                    rf"(?<!\w){escaped}(?!\w)"
                )
                text, prop_count = propagated_pattern.subn("[NAME]", text)
                count += prop_count

        return text, count

    def _has_phone_match(self, text: str) -> bool:
        for match in _LABELED_PHONE_PATTERN.finditer(text):
            if 7 <= _count_digits(match.group(2)) <= 15:
                return True
        for match in _UNLABELED_PHONE_PATTERN.finditer(text):
            value: str = match.group(0)
            min_digits: int = 8 if value.startswith("+") else 9
            if min_digits <= _count_digits(value) <= 15:
                return True
        return False

    def _has_name_match(self, text: str) -> bool:
        for match in _LABELED_PATIENT_NAME_PATTERN.finditer(text):
            candidate, _ = _trim_clinical_suffix(match.group(2))
            if _is_valid_name_candidate(candidate, allow_stopword_single=False):
                return True
        for match in _HONORIFIC_NAME_PATTERN.finditer(text):
            candidate, _ = _trim_clinical_suffix(match.group(2))
            if _is_valid_name_candidate(candidate, allow_stopword_single=True):
                return True
        return False


def deidentify_text(
    text: str,
    config: DeidentifierConfig | None = None,
) -> DeidentificationResult:
    """De-identify a clinical text string using HIPAA Safe Harbor rules."""
    return HIPAADeidentifier(config=config).deidentify(text)
