"""Identifier conventions of widely used clinical databases and data models.

A :class:`DatasetProfile` records which columns identify a patient, an
encounter, a unit stay and a care site in a given data source. Audits use
these to choose the clustering unit for the bootstrap (resample *patients*,
not rows, when patients contribute several stays) and the site column for
site-level summaries. Profiles describe structure only; they do not define
cohorts or outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    patient_id: str
    encounter_id: Optional[str] = None
    stay_id: Optional[str] = None
    site_id: Optional[str] = None
    site_table: Optional[str] = None
    modules: Dict[str, List[str]] = field(default_factory=dict)
    notes: str = ""

    def cluster_column(self) -> str:
        """Column to pass as ``groups`` for a cluster bootstrap."""
        return self.patient_id

    def module_of(self, table: str) -> Optional[str]:
        for module, tables in self.modules.items():
            if table.lower() in (t.lower() for t in tables):
                return module
        return None


MIMIC_IV = DatasetProfile(
    name="MIMIC-IV",
    patient_id="subject_id",
    encounter_id="hadm_id",
    stay_id="stay_id",
    site_id=None,
    modules={
        "hosp": [
            "admissions", "patients", "transfers", "services", "labevents", "d_labitems", "microbiologyevents",
            "diagnoses_icd", "d_icd_diagnoses", "procedures_icd", "d_icd_procedures", "drgcodes", "hcpcsevents",
            "d_hcpcs", "emar", "emar_detail", "pharmacy", "poe", "poe_detail", "prescriptions", "omr", "provider",
        ],
        "icu": [
            "icustays", "chartevents", "d_items", "datetimeevents", "inputevents", "ingredientevents",
            "outputevents", "procedureevents", "caregiver",
        ],
    },
    notes="Single academic centre; patients can have several admissions and ICU stays, so cluster on subject_id.",
)

EICU_CRD = DatasetProfile(
    name="eICU-CRD",
    patient_id="uniquepid",
    encounter_id="patienthealthsystemstayid",
    stay_id="patientunitstayid",
    site_id="hospitalid",
    site_table="patient",
    modules={
        "": [
            "patient", "hospital", "vitalPeriodic", "vitalAperiodic", "lab", "customLab", "medication",
            "infusionDrug", "intakeOutput", "nurseCharting", "nurseCare", "nurseAssessment", "physicalExam",
            "respiratoryCare", "respiratoryCharting", "treatment", "diagnosis", "admissionDx", "admissionDrug",
            "pastHistory", "allergy", "microLab", "note", "carePlanGeneral", "carePlanGoal", "carePlanCareProvider",
            "carePlanEOL", "carePlanInfectiousDisease", "apacheApsVar", "apachePredVar", "apachePatientResult",
        ]
    },
    notes="Multi-centre; most event tables are keyed by patientunitstayid and the patient table maps stays to "
    "uniquepid and hospitalid. Event times are offsets in minutes from unit admission.",
)

MIMIC_III = DatasetProfile(
    name="MIMIC-III",
    patient_id="SUBJECT_ID",
    encounter_id="HADM_ID",
    stay_id="ICUSTAY_ID",
    notes="Column names are upper case in the distributed CSV files.",
)

OMOP_CDM = DatasetProfile(
    name="OMOP CDM",
    patient_id="person_id",
    encounter_id="visit_occurrence_id",
    stay_id="visit_detail_id",
    site_id="care_site_id",
    site_table="visit_occurrence",
    modules={"": ["person", "visit_occurrence", "visit_detail", "measurement", "observation", "condition_occurrence",
                  "drug_exposure", "procedure_occurrence", "death", "care_site", "concept"]},
)

MEDS = DatasetProfile(
    name="MEDS",
    patient_id="subject_id",
    notes="Medical Event Data Standard: long-format events with subject_id, time, code and optional values.",
)

PROFILES: Dict[str, DatasetProfile] = {
    "mimic-iv": MIMIC_IV,
    "eicu-crd": EICU_CRD,
    "mimic-iii": MIMIC_III,
    "omop": OMOP_CDM,
    "meds": MEDS,
}


def get_profile(name: str) -> DatasetProfile:
    compact = "".join(ch for ch in name.lower() if ch.isalnum())
    aliases = {
        "mimiciv": "mimic-iv", "mimic4": "mimic-iv",
        "eicu": "eicu-crd", "eicucrd": "eicu-crd",
        "mimiciii": "mimic-iii", "mimic3": "mimic-iii",
        "omop": "omop", "omopcdm": "omop",
        "meds": "meds",
    }
    if compact not in aliases:
        raise KeyError(f"Unknown profile {name!r}. Available: {sorted(PROFILES)}.")
    return PROFILES[aliases[compact]]
