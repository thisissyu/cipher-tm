"""Public core-method API for CIPHER-TM.

This package exposes only the paper's algorithmic primitives.  Data handling,
providers, prompts, experiment configuration, evaluation, and orchestration are
intentionally outside the reference release.
"""

from .adjudication import (
    CandidateRankJudgment,
    RankRepair,
    ResolvedJudgment,
    TopicContract,
    apply_soft_anchors,
    normalized_entropy,
    resolve_candidate_rank,
    select_entropy_queries,
    top_candidate_topics,
    validate_and_resolve_judgments,
)
from .decoder import (
    DecodedTopicWords,
    candidate_npmi,
    decode_topic_words,
    weighted_topic_term_scores,
)
from .refinement import (
    cosine_knn_transition,
    posterior_weighted_prototypes,
    propagate_posteriors,
    prototype_feedback,
    row_normalize,
)

__all__ = [
    "CandidateRankJudgment",
    "DecodedTopicWords",
    "RankRepair",
    "ResolvedJudgment",
    "TopicContract",
    "apply_soft_anchors",
    "candidate_npmi",
    "cosine_knn_transition",
    "decode_topic_words",
    "normalized_entropy",
    "posterior_weighted_prototypes",
    "propagate_posteriors",
    "prototype_feedback",
    "resolve_candidate_rank",
    "row_normalize",
    "select_entropy_queries",
    "top_candidate_topics",
    "validate_and_resolve_judgments",
    "weighted_topic_term_scores",
]
