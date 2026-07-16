from social_evaluation.metrics_evaluation.metrics import metric
from social_evaluation.metrics_evaluation.metrics_utils import metrics as metrics_dict
import social_evaluation.metrics_evaluation.fleet  # noqa: F401  registers fleet metrics

__all__ = ["metric", "metrics_dict"]
