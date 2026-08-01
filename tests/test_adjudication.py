from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cipher_tm.adjudication import (
    CandidateRankJudgment,
    ResolvedJudgment,
    TopicContract,
    apply_soft_anchors,
    normalized_entropy,
    resolve_candidate_rank,
    select_entropy_queries,
    top_candidate_topics,
    validate_and_resolve_judgments,
)


def test_topic_contract_is_deeply_immutable() -> None:
    cues = ["spacecraft", "orbit"]
    contract = TopicContract(
        topic_id=3,
        short_name="space flight",
        definition="Documents centrally concerned with space flight.",
        positive_cues=cues,
        exclusions=["computer graphics"],
        confusable_with=[7],
    )
    cues.append("mutated later")

    assert contract.positive_cues == ("spacecraft", "orbit")
    assert contract.exclusions == ("computer graphics",)
    assert contract.confusable_with == (7,)
    with pytest.raises(FrozenInstanceError):
        contract.topic_id = 4  # type: ignore[misc]


def test_entropy_selection_and_local_candidates_follow_posterior_geometry() -> None:
    posterior = np.array(
        [
            [0.90, 0.05, 0.05],
            [1 / 3, 1 / 3, 1 / 3],
            [0.15, 0.55, 0.30],
        ]
    )

    entropy = normalized_entropy(posterior)
    selected = select_entropy_queries(posterior, budget=2)
    candidates = top_candidate_topics(posterior, candidate_count=2)

    assert entropy[1] == pytest.approx(1.0)
    assert selected.tolist() == [1, 2]
    assert candidates.tolist() == [[0, 1], [0, 1], [1, 2]]


def test_entropy_query_ties_are_broken_by_document_index() -> None:
    posterior = np.full((3, 2), 0.5)

    assert select_entropy_queries(posterior, budget=10).tolist() == [0, 1, 2]
    assert select_entropy_queries(posterior, budget=-1).tolist() == []


@pytest.mark.parametrize(
    "posterior, message",
    [
        ([0.4, 0.6], "shape"),
        ([[0.4, 0.4]], "sum to one"),
        ([[1.1, -0.1]], "non-negative"),
        ([[np.nan, np.nan]], "finite"),
    ],
)
def test_posterior_validation_is_explicit(posterior: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_entropy_queries(posterior, budget=1)


def _candidate_judgment(
    doc_id: str,
    rank: int,
    *,
    alternative: int | None = None,
    confidence: float = 0.9,
) -> CandidateRankJudgment:
    return CandidateRankJudgment(
        doc_id=doc_id,
        candidate_rank=rank,
        alternative_candidate_rank=alternative,
        confidence=confidence,
        rationale="contract-grounded decision",
    )


def test_candidate_rank_is_resolved_in_each_documents_local_space() -> None:
    resolved, repair = resolve_candidate_rank(
        _candidate_judgment("doc-b", 1, alternative=2),
        [9, 4, 7],
    )

    assert resolved.topic_id == 4
    assert resolved.alternative_topic_id == 7
    assert resolved.confidence == 0.9
    assert repair is None


def test_invalid_primary_rank_uses_valid_alternative_and_caps_confidence() -> None:
    resolved, repair = resolve_candidate_rank(
        _candidate_judgment("doc-a", 99, alternative=2),
        [9, 4, 7],
    )

    assert resolved.topic_id == 7
    assert resolved.confidence == 0.5
    assert repair is not None
    assert repair.policy == "valid_alternative_candidate_rank"
    assert repair.resolved_candidate_rank == 2


def test_invalid_ranks_fall_back_to_highest_prior_numerical_candidate() -> None:
    resolved, repair = resolve_candidate_rank(
        _candidate_judgment("doc-a", -3, alternative=8),
        [9, 4, 7],
    )

    assert resolved.topic_id == 9
    assert resolved.alternative_topic_id == 4
    assert repair is not None
    assert repair.policy == "top_numerical_candidate_rank"
    assert repair.resolved_candidate_rank == 0


def test_identity_validation_reorders_results_and_rejects_mismatches() -> None:
    judgments = [
        _candidate_judgment("doc-b", 1),
        _candidate_judgment("doc-a", 0),
    ]
    resolved, repairs = validate_and_resolve_judgments(
        judgments,
        requested_doc_ids=["doc-a", "doc-b"],
        candidates_by_doc_id={"doc-a": [7, 8], "doc-b": [2, 5]},
    )

    assert [(item.doc_id, item.topic_id) for item in resolved] == [("doc-a", 7), ("doc-b", 5)]
    assert repairs == ()

    with pytest.raises(ValueError, match=r"missing=\['doc-a'\].*unexpected=\['doc-x'\]"):
        validate_and_resolve_judgments(
            [_candidate_judgment("doc-x", 0)],
            requested_doc_ids=["doc-a"],
            candidates_by_doc_id={"doc-a": [7, 8]},
        )


def test_duplicate_response_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate judgment document IDs"):
        validate_and_resolve_judgments(
            [_candidate_judgment("doc-a", 0), _candidate_judgment("doc-a", 1)],
            requested_doc_ids=["doc-a"],
            candidates_by_doc_id={"doc-a": [7, 8]},
        )


def test_requested_document_ids_reject_a_bare_string() -> None:
    with pytest.raises(TypeError, match="sequence of document IDs"):
        validate_and_resolve_judgments(
            [_candidate_judgment("doc-a", 0)],
            requested_doc_ids="doc-a",  # type: ignore[arg-type]
            candidates_by_doc_id={"doc-a": [7, 8]},
        )


def test_soft_anchor_uses_confidence_floor_and_explicit_identity_mapping() -> None:
    posterior = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    judgments = [
        ResolvedJudgment(doc_id="low", topic_id=1, confidence=0.0),
        ResolvedJudgment(doc_id="high", topic_id=0, confidence=0.8),
    ]

    anchored = apply_soft_anchors(
        posterior,
        judgments,
        {"high": 1, "low": 0},
        anchor_weight=0.9,
        minimum_confidence=0.1,
    )

    # low: eta=.09 -> .91 * [.7, .3] + .09 * [0, 1]
    assert anchored[0] == pytest.approx([0.637, 0.363])
    # high: eta=.72 -> .28 * [.2, .8] + .72 * [1, 0]
    assert anchored[1] == pytest.approx([0.776, 0.224])
    assert anchored[2] == pytest.approx(posterior[2])
    assert posterior.tolist() == [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]


def test_soft_anchor_rejects_unmapped_documents_and_out_of_range_topics() -> None:
    posterior = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="no posterior row"):
        apply_soft_anchors(
            posterior,
            [ResolvedJudgment("unknown", 0, 0.8)],
            {},
            anchor_weight=0.9,
        )
    with pytest.raises(ValueError, match="topic_id 2.*out of bounds"):
        apply_soft_anchors(
            posterior,
            [ResolvedJudgment("doc-a", 2, 0.8)],
            {"doc-a": 0},
            anchor_weight=0.9,
        )
