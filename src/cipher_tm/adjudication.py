"""Core primitives for candidate-constrained semantic adjudication.

The functions in this module implement the paper's bounded semantic correction:
numerical posteriors determine both the queried documents and each document's
allowed topics, while semantic judgments can only soften those posteriors toward
one of the allowed candidates.  Provider calls, prompts, batching, persistence,
and experiment orchestration are intentionally outside this public core.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

RepairPolicy = Literal[
    "valid_alternative_candidate_rank",
    "top_numerical_candidate_rank",
]


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        return int(operator.index(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _string_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"{name} must contain only strings")
    return result


@dataclass(frozen=True, slots=True)
class TopicContract:
    """A discriminative semantic specification for one fixed numerical topic.

    Contracts describe supporting evidence, exclusions, and nearby confounders;
    they never create or renumber topics.  Sequence fields are normalized to
    tuples so instances remain immutable even when callers pass lists.
    """

    topic_id: int
    short_name: str
    definition: str
    positive_cues: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    confusable_with: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        topic_id = _integer(self.topic_id, name="topic_id")
        if topic_id < 0:
            raise ValueError("topic_id must be non-negative")
        if not isinstance(self.short_name, str) or not self.short_name.strip():
            raise ValueError("short_name must be a non-empty string")
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError("definition must be a non-empty string")

        positive_cues = _string_tuple(self.positive_cues, name="positive_cues")
        exclusions = _string_tuple(self.exclusions, name="exclusions")
        confusable_with = tuple(
            _integer(value, name="confusable_with item") for value in self.confusable_with
        )
        if any(topic < 0 for topic in confusable_with):
            raise ValueError("confusable topic IDs must be non-negative")

        object.__setattr__(self, "topic_id", topic_id)
        object.__setattr__(self, "positive_cues", positive_cues)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "confusable_with", confusable_with)


@dataclass(frozen=True, slots=True)
class CandidateRankJudgment:
    """A semantic decision expressed in a document-local candidate space.

    Candidate ranks are deliberately not range-checked here: the relevant
    candidate list belongs to the document and is supplied when resolving the
    judgment.  This also permits deterministic repair of malformed responses.
    """

    doc_id: str
    candidate_rank: int
    confidence: float
    alternative_candidate_rank: int | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be a non-empty string")
        candidate_rank = _integer(self.candidate_rank, name="candidate_rank")
        alternative = self.alternative_candidate_rank
        if alternative is not None:
            alternative = _integer(alternative, name="alternative_candidate_rank")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as error:
            raise TypeError("confidence must be a real number") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and lie in [0, 1]")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")

        object.__setattr__(self, "candidate_rank", candidate_rank)
        object.__setattr__(self, "alternative_candidate_rank", alternative)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class ResolvedJudgment:
    """A validated judgment mapped from a local rank to a global topic ID."""

    doc_id: str
    topic_id: int
    confidence: float
    alternative_topic_id: int | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be a non-empty string")
        topic_id = _integer(self.topic_id, name="topic_id")
        if topic_id < 0:
            raise ValueError("topic_id must be non-negative")
        alternative = self.alternative_topic_id
        if alternative is not None:
            alternative = _integer(alternative, name="alternative_topic_id")
            if alternative < 0:
                raise ValueError("alternative_topic_id must be non-negative")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as error:
            raise TypeError("confidence must be a real number") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and lie in [0, 1]")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")

        object.__setattr__(self, "topic_id", topic_id)
        object.__setattr__(self, "alternative_topic_id", alternative)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class RankRepair:
    """Audit information for a deterministic candidate-rank repair."""

    doc_id: str
    policy: RepairPolicy
    original_candidate_rank: int
    original_alternative_candidate_rank: int | None
    resolved_candidate_rank: int
    resolved_topic_id: int


def _posterior_array(
    posterior: ArrayLike,
    *,
    minimum_topics: int = 1,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(posterior, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("posterior must be a rectangular numeric matrix") from error
    if values.ndim != 2:
        raise ValueError("posterior must have shape (documents, topics)")
    if values.shape[1] < minimum_topics:
        raise ValueError(f"posterior must contain at least {minimum_topics} topic(s)")
    if not np.all(np.isfinite(values)):
        raise ValueError("posterior values must be finite")
    if np.any(values < 0.0):
        raise ValueError("posterior values must be non-negative")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("each posterior row must sum to one")
    return values


def normalized_entropy(posterior: ArrayLike, *, epsilon: float = 1e-12) -> NDArray[np.float64]:
    """Return each document's entropy normalized by ``log(number_of_topics)``.

    The input must be a finite, non-negative row-stochastic matrix with at least
    two topics.  The small positive ``epsilon`` mirrors the paper's numerical
    safeguard inside the logarithm.
    """

    values = _posterior_array(posterior, minimum_topics=2)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive finite number")
    topic_count = values.shape[1]
    return -np.sum(values * np.log(values + epsilon), axis=1) / math.log(topic_count)


def top_candidate_topics(posterior: ArrayLike, candidate_count: int) -> NDArray[np.int64]:
    """Return the document-local Top-m global topic IDs in descending prior order.

    ``candidate_count`` is capped at the topic count.  Stable sorting makes ties
    deterministic: lower global topic IDs precede higher IDs with equal mass.
    """

    values = _posterior_array(posterior)
    count = _integer(candidate_count, name="candidate_count")
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    count = min(count, values.shape[1])
    return np.argsort(-values, axis=1, kind="stable")[:, :count].astype(np.int64, copy=False)


def select_entropy_queries(posterior: ArrayLike, budget: int) -> NDArray[np.int64]:
    """Select up to ``budget`` document rows with highest normalized entropy.

    A non-positive budget selects no documents, while a budget larger than the
    corpus selects every document.  Equal-entropy rows are ordered by their
    original document index for reproducibility.
    """

    values = _posterior_array(posterior, minimum_topics=2)
    query_budget = _integer(budget, name="budget")
    count = min(max(query_budget, 0), values.shape[0])
    if count == 0:
        return np.empty(0, dtype=np.int64)
    entropy = normalized_entropy(values)
    order = np.lexsort((np.arange(values.shape[0]), -entropy))
    return order[:count].astype(np.int64, copy=False)


def _candidate_ids(candidate_topic_ids: Sequence[int], *, doc_id: str) -> tuple[int, ...]:
    if isinstance(candidate_topic_ids, (str, bytes)):
        raise TypeError(f"candidate topics for {doc_id!r} must be a sequence of integers")
    result = tuple(
        _integer(topic_id, name=f"candidate topic ID for {doc_id!r}")
        for topic_id in candidate_topic_ids
    )
    if not result:
        raise ValueError(f"candidate topics for {doc_id!r} must not be empty")
    if any(topic_id < 0 for topic_id in result):
        raise ValueError(f"candidate topics for {doc_id!r} must be non-negative")
    if len(result) != len(set(result)):
        raise ValueError(f"candidate topics for {doc_id!r} must be unique")
    return result


def resolve_candidate_rank(
    judgment: CandidateRankJudgment,
    candidate_topic_ids: Sequence[int],
    *,
    repaired_confidence_cap: float = 0.5,
) -> tuple[ResolvedJudgment, RankRepair | None]:
    """Resolve a local candidate rank, deterministically repairing invalid ranks.

    A valid primary rank is used directly.  Otherwise a valid alternative rank
    is used; if neither is valid, rank zero (the highest-prior numerical
    candidate) is selected.  Any repair caps confidence at
    ``repaired_confidence_cap`` and returns an immutable audit record.
    """

    if not isinstance(judgment, CandidateRankJudgment):
        raise TypeError("judgment must be a CandidateRankJudgment")
    candidates = _candidate_ids(candidate_topic_ids, doc_id=judgment.doc_id)
    try:
        confidence_cap = float(repaired_confidence_cap)
    except (TypeError, ValueError) as error:
        raise TypeError("repaired_confidence_cap must be a real number") from error
    if not math.isfinite(confidence_cap) or not 0.0 <= confidence_cap <= 1.0:
        raise ValueError("repaired_confidence_cap must be finite and lie in [0, 1]")

    primary_rank = judgment.candidate_rank
    alternative_rank = judgment.alternative_candidate_rank
    primary_valid = 0 <= primary_rank < len(candidates)
    valid_alternative_rank = (
        alternative_rank
        if alternative_rank is not None and 0 <= alternative_rank < len(candidates)
        else None
    )

    policy: RepairPolicy | None = None
    if primary_valid:
        resolved_rank = primary_rank
    elif valid_alternative_rank is not None:
        resolved_rank = valid_alternative_rank
        policy = "valid_alternative_candidate_rank"
    else:
        resolved_rank = 0
        policy = "top_numerical_candidate_rank"

    alternative_topic_id: int | None
    if valid_alternative_rank is not None and valid_alternative_rank != resolved_rank:
        alternative_topic_id = candidates[valid_alternative_rank]
    else:
        alternative_topic_id = next(
            (topic_id for rank, topic_id in enumerate(candidates) if rank != resolved_rank),
            None,
        )

    repair: RankRepair | None = None
    confidence = judgment.confidence
    rationale = judgment.rationale
    if policy is not None:
        confidence = min(confidence, confidence_cap)
        rationale = (
            f"{rationale} [candidate-rank constraint repaired from rank "
            f"{primary_rank} to {resolved_rank} using {policy}]"
        ).strip()
        repair = RankRepair(
            doc_id=judgment.doc_id,
            policy=policy,
            original_candidate_rank=primary_rank,
            original_alternative_candidate_rank=alternative_rank,
            resolved_candidate_rank=resolved_rank,
            resolved_topic_id=candidates[resolved_rank],
        )

    return (
        ResolvedJudgment(
            doc_id=judgment.doc_id,
            topic_id=candidates[resolved_rank],
            confidence=confidence,
            alternative_topic_id=alternative_topic_id,
            rationale=rationale,
        ),
        repair,
    )


def validate_and_resolve_judgments(
    judgments: Iterable[CandidateRankJudgment],
    requested_doc_ids: Sequence[str],
    candidates_by_doc_id: Mapping[str, Sequence[int]],
) -> tuple[tuple[ResolvedJudgment, ...], tuple[RankRepair, ...]]:
    """Validate response identities and resolve judgments in request order.

    Every requested document must occur exactly once, and unexpected or
    duplicate identities are rejected.  Candidate lists are document-local;
    consequently, the same returned rank may map to different global topics.
    The function performs no provider retries or other orchestration.
    """

    if isinstance(requested_doc_ids, (str, bytes)):
        raise TypeError("requested_doc_ids must be a sequence of document IDs")
    requested = tuple(requested_doc_ids)
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in requested):
        raise ValueError("requested_doc_ids must contain non-empty strings")
    requested_duplicates = sorted(
        doc_id for doc_id, count in Counter(requested).items() if count > 1
    )
    if requested_duplicates:
        raise ValueError(f"requested document IDs must be unique: {requested_duplicates}")

    received = tuple(judgments)
    if any(not isinstance(judgment, CandidateRankJudgment) for judgment in received):
        raise TypeError("judgments must contain only CandidateRankJudgment instances")
    received_ids = [judgment.doc_id for judgment in received]
    duplicates = sorted(doc_id for doc_id, count in Counter(received_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate judgment document IDs: {duplicates}")

    requested_set = set(requested)
    received_set = set(received_ids)
    missing = [doc_id for doc_id in requested if doc_id not in received_set]
    unexpected = sorted(received_set - requested_set)
    if missing or unexpected:
        raise ValueError(
            "judgment document IDs do not match request: "
            f"missing={missing}, unexpected={unexpected}"
        )
    missing_candidates = [doc_id for doc_id in requested if doc_id not in candidates_by_doc_id]
    if missing_candidates:
        raise ValueError(f"candidate lists are missing for documents: {missing_candidates}")

    by_id = {judgment.doc_id: judgment for judgment in received}
    resolved: list[ResolvedJudgment] = []
    repairs: list[RankRepair] = []
    for doc_id in requested:
        item, repair = resolve_candidate_rank(by_id[doc_id], candidates_by_doc_id[doc_id])
        resolved.append(item)
        if repair is not None:
            repairs.append(repair)
    return tuple(resolved), tuple(repairs)


def apply_soft_anchors(
    posterior: ArrayLike,
    judgments: Iterable[ResolvedJudgment],
    document_indices: Mapping[str, int],
    *,
    anchor_weight: float,
    minimum_confidence: float = 0.1,
) -> NDArray[np.float64]:
    r"""Apply confidence-weighted posterior anchors without parsing document IDs.

    For each judgment this implements
    :math:`\eta_i=\min(1,\lambda\max(u_{min},u_i))` and
    :math:`\widetilde q_i=(1-\eta_i)q_i+\eta_i e_{\hat k_i}`.
    ``document_indices`` explicitly maps stable identities to posterior rows;
    unjudged rows are unchanged.
    """

    values = _posterior_array(posterior).copy()
    try:
        weight = float(anchor_weight)
    except (TypeError, ValueError) as error:
        raise TypeError("anchor_weight must be a real number") from error
    try:
        confidence_floor = float(minimum_confidence)
    except (TypeError, ValueError) as error:
        raise TypeError("minimum_confidence must be a real number") from error
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("anchor_weight must be a finite non-negative number")
    if not math.isfinite(confidence_floor) or not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("minimum_confidence must be finite and lie in [0, 1]")

    items = tuple(judgments)
    if any(not isinstance(judgment, ResolvedJudgment) for judgment in items):
        raise TypeError("judgments must contain only ResolvedJudgment instances")
    duplicate_ids = sorted(
        doc_id for doc_id, count in Counter(item.doc_id for item in items).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate anchored document IDs: {duplicate_ids}")

    used_rows: dict[int, str] = {}
    for judgment in items:
        if judgment.doc_id not in document_indices:
            raise ValueError(f"no posterior row is mapped for document {judgment.doc_id!r}")
        index = _integer(document_indices[judgment.doc_id], name="document row index")
        if not 0 <= index < values.shape[0]:
            raise ValueError(f"posterior row index for {judgment.doc_id!r} is out of bounds")
        if index in used_rows:
            raise ValueError(
                f"documents {used_rows[index]!r} and {judgment.doc_id!r} map to the same row"
            )
        used_rows[index] = judgment.doc_id
        if not 0 <= judgment.topic_id < values.shape[1]:
            raise ValueError(
                f"topic_id {judgment.topic_id} for {judgment.doc_id!r} is out of bounds"
            )

        eta = min(1.0, weight * max(confidence_floor, judgment.confidence))
        values[index] *= 1.0 - eta
        values[index, judgment.topic_id] += eta

    # The convex update is row-stochastic analytically.  This normalization only
    # removes floating-point drift and does not alter the anchor equation.
    row_sums = values.sum(axis=1, keepdims=True)
    return values / row_sums


__all__ = [
    "CandidateRankJudgment",
    "RankRepair",
    "RepairPolicy",
    "ResolvedJudgment",
    "TopicContract",
    "apply_soft_anchors",
    "normalized_entropy",
    "resolve_candidate_rank",
    "select_entropy_queries",
    "top_candidate_topics",
    "validate_and_resolve_judgments",
]
