try:
    from .evaluator import EvaluationPipeline
    __all__ = ["EvaluationPipeline", "HateCommunityEvaluationPipeline"]
except ImportError:
    __all__ = ["HateCommunityEvaluationPipeline"]

from .hate_evaluator import HateCommunityEvaluationPipeline

