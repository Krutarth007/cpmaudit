"""cpmaudit: audit the reliability of individual predictions from clinical
risk models.

Main entry points
-----------------
StabilityAudit / stability_audit
    Bootstrap the full model-building process and measure how much each
    individual's predicted risk and decision depend on the development sample.
MultiplicityAudit
    Measure disagreement between equally well-performing models.
instability_curve
    Instability of individual predictions as a function of development size.
minimum_sample_size_binary
    Analytic minimum sample size for a binary-outcome risk model.
StabilityPolicy
    Pre-agreed acceptance limits evaluated against an audit.
io.TableStore
    Discover and read tables from local clinical data releases.
"""

from ._version import __version__
from .calibration import calibration_curve, calibration_error_summary
from .governance import PolicyReport, StabilityPolicy, model_facts
from .learning_curve import InstabilityCurve, instability_curve
from .metrics import performance_summary
from .models import ModelBuilder, make_model
from .multiplicity import (
    MultiplicityAudit,
    MultiplicityResult,
    multiplicity_from_predictions,
    multiplicity_measures,
    param_grid_variants,
    seed_variants,
)
from .sampsize import cox_snell_r2_from_c, max_cox_snell_r2, minimum_sample_size_binary
from .stability import StabilityAudit, StabilityResult, stability_audit

__all__ = [
    "__version__",
    "StabilityAudit",
    "StabilityResult",
    "stability_audit",
    "MultiplicityAudit",
    "MultiplicityResult",
    "multiplicity_from_predictions",
    "multiplicity_measures",
    "seed_variants",
    "param_grid_variants",
    "instability_curve",
    "InstabilityCurve",
    "minimum_sample_size_binary",
    "cox_snell_r2_from_c",
    "max_cox_snell_r2",
    "StabilityPolicy",
    "PolicyReport",
    "model_facts",
    "ModelBuilder",
    "make_model",
    "performance_summary",
    "calibration_curve",
    "calibration_error_summary",
]
