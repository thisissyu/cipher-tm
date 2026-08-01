"""Posterior-aware, globally constrained topic-word decoding.

This module contains only the lexical decoding mechanism described in the
paper.  It deliberately has no dependency on data loading, experiment
configuration, model providers, evaluation, or pipeline orchestration.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class DecodedTopicWords:
    """The words selected for every topic, with their vocabulary indices."""

    words: tuple[tuple[str, ...], ...]
    indices: tuple[tuple[int, ...], ...]


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def weighted_topic_term_scores(
    document_term: sp.spmatrix | np.ndarray,
    posteriors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior-weighted salience and exclusivity.

    For document-term counts ``X`` and document-topic posteriors ``Q``, the
    soft topic-term counts are ``N = Q.T @ X``.  The returned arrays are, in
    order, the paper's salience ``s_kv``, exclusivity ``e_kv``, and soft
    counts ``n_kv``.  Their common shape is ``(n_topics, vocabulary_size)``.

    The function accepts soft posteriors; hard one-hot assignments are only a
    special case.  Every posterior row must lie on the probability simplex, as
    required by the paper's definition of ``Q``, and every topic must receive
    positive total posterior mass before it can be assigned descriptor words.
    """

    bow, probabilities = _validate_matrices(document_term, posteriors)

    weighted = np.asarray(probabilities.T @ bow, dtype=np.float64)
    topic_totals = np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)
    word_given_topic = weighted / topic_totals

    global_counts = np.asarray(bow.sum(axis=0)).ravel().astype(np.float64)
    global_probability = global_counts / max(float(global_counts.sum()), 1e-12)
    lift = word_given_topic / np.maximum(global_probability[None, :], 1e-12)

    salience = np.log1p(weighted) * np.log1p(lift)
    exclusivity = weighted / np.maximum(weighted.sum(axis=0, keepdims=True), 1e-12)
    return salience, exclusivity, weighted


def candidate_npmi(
    document_term: sp.spmatrix | np.ndarray,
    candidate_indices: Sequence[int],
) -> np.ndarray:
    """Return document-level NPMI for a local vocabulary candidate pool.

    Counts are binarized before document frequencies and co-occurrences are
    computed.  Unobserved pairs receive ``-1`` and the diagonal is ``1``.
    The candidate order is preserved in both output dimensions.
    """

    bow = _as_count_matrix(document_term)
    if isinstance(candidate_indices, (str, bytes)):
        raise ValueError("candidate_indices must be a sequence of integers")
    raw_candidates = np.asarray(candidate_indices, dtype=object)
    if raw_candidates.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    candidate_values = [
        _integer(value, name="candidate index", minimum=0) for value in raw_candidates.tolist()
    ]
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("candidate_indices must not contain duplicates")
    if any(candidate >= bow.shape[1] for candidate in candidate_values):
        raise ValueError("candidate index is outside the vocabulary")
    candidates = np.asarray(candidate_values, dtype=np.int64)

    binary = (bow > 0).astype(np.int8).tocsc()
    frequencies = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
    if np.any(frequencies[candidates] == 0.0):
        raise ValueError("candidate index refers to an unobserved vocabulary term")
    return _candidate_npmi(binary, candidates, frequencies)


def decode_topic_words(
    document_term: sp.spmatrix | np.ndarray,
    posteriors: np.ndarray,
    vocabulary: Sequence[str],
    *,
    top_n: int = 15,
    candidate_pool: int = 220,
    beta: float = 0.35,
    gamma: float = 0.45,
    exclusivity_scale: float = 1_000.0,
    global_unique: bool = True,
) -> DecodedTopicWords:
    """Decode concise topic descriptors from posterior-weighted word counts.

    Candidate words are ranked by a convex combination of salience and
    exclusivity.  Selection then adds mean candidate-pool NPMI to favor words
    that co-occur with those already chosen for the topic.  Topics take turns
    selecting one word at each rank, so a hard global non-reuse constraint
    cannot systematically starve later topics.

    When ``global_unique`` is enabled, each local pool is expanded to at least
    ``n_topics * top_n`` candidates.  This makes the round-robin allocation
    feasible whenever the corpus contains enough distinct observed words.
    Ties are resolved by ascending vocabulary index for reproducible output.

    Args:
        document_term: Non-negative document-by-term count matrix.
        posteriors: Row-stochastic document-by-topic posterior matrix with positive mass
            for every topic.
        vocabulary: Unique word at each column of ``document_term``.
        top_n: Number of descriptor words selected per topic.
        candidate_pool: Nominal number of locally ranked candidates per topic.
        beta: Exclusivity weight in the base lexical score, in ``[0, 1]``.
        gamma: Weight of mean pairwise NPMI during greedy selection.
        exclusivity_scale: The paper's ``kappa`` inside ``log(1 + kappa e)``.
        global_unique: If true, prohibit reuse of a word by another topic.

    Returns:
        Immutable words and vocabulary indices in topic order.

    Raises:
        ValueError: If shapes, values, parameters, or uniqueness feasibility
            are invalid.
    """

    bow, probabilities = _validate_matrices(document_term, posteriors)
    if isinstance(vocabulary, (str, bytes)):
        raise ValueError("vocabulary must be a sequence of strings")
    vocab = tuple(vocabulary)
    if len(vocab) != bow.shape[1]:
        raise ValueError(
            f"vocabulary length must match document_term columns: {len(vocab)} != {bow.shape[1]}"
        )
    if any(not isinstance(word, str) or not word for word in vocab):
        raise ValueError("vocabulary must contain non-empty strings")
    if len(set(vocab)) != len(vocab):
        raise ValueError("vocabulary terms must be unique")

    top_n = _integer(top_n, name="top_n", minimum=1)
    candidate_pool = _integer(candidate_pool, name="candidate_pool", minimum=1)
    beta = _finite_float(beta, name="beta")
    gamma = _finite_float(gamma, name="gamma")
    exclusivity_scale = _finite_float(exclusivity_scale, name="exclusivity_scale")
    if not isinstance(global_unique, (bool, np.bool_)):
        raise ValueError("global_unique must be a boolean")
    global_unique = bool(global_unique)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    if gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if exclusivity_scale <= 0.0:
        raise ValueError("exclusivity_scale must be finite and positive")

    topic_count = int(probabilities.shape[1])
    global_counts = np.asarray(bow.sum(axis=0)).ravel()
    active_word_ids = np.flatnonzero(global_counts > 0.0).astype(np.int64, copy=False)
    active_vocabulary_size = int(active_word_ids.size)
    required_unique_words = topic_count * top_n

    if top_n > active_vocabulary_size:
        raise ValueError(
            "cannot select observed distinct words within each topic: "
            f"top_n={top_n}, observed_vocabulary_size={active_vocabulary_size}"
        )
    if global_unique and active_vocabulary_size < required_unique_words:
        raise ValueError(
            "global topic-word uniqueness is impossible: "
            f"observed_vocabulary_size={active_vocabulary_size}, "
            f"required={required_unique_words} "
            f"({topic_count} topics x {top_n} words)"
        )

    salience, exclusivity, _ = weighted_topic_term_scores(bow, probabilities)
    base_score = (1.0 - beta) * salience + beta * np.log1p(exclusivity_scale * exclusivity)

    effective_pool = min(
        active_vocabulary_size,
        max(candidate_pool, required_unique_words if global_unique else top_n),
    )
    candidates = tuple(
        active_word_ids[
            np.lexsort((active_word_ids, -base_score[topic_id, active_word_ids]))[:effective_pool]
        ]
        for topic_id in range(topic_count)
    )

    binary = (bow > 0).astype(np.int8).tocsc()
    document_frequencies = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
    npmi = tuple(
        _candidate_npmi(binary, topic_candidates, document_frequencies)
        for topic_candidates in candidates
    )

    positions = tuple(
        {int(word_id): local_id for local_id, word_id in enumerate(topic_candidates)}
        for topic_candidates in candidates
    )
    selected: list[list[int]] = [[] for _ in range(topic_count)]
    globally_used: set[int] = set()

    # Rank-major traversal is the round-robin allocation described in the paper.
    for _rank in range(top_n):
        for topic_id in range(topic_count):
            chosen_word: int | None = None
            chosen_score = -np.inf
            selected_positions = [positions[topic_id][word_id] for word_id in selected[topic_id]]

            for local_id, raw_word_id in enumerate(candidates[topic_id]):
                word_id = int(raw_word_id)
                if word_id in selected[topic_id]:
                    continue
                if global_unique and word_id in globally_used:
                    continue

                coherence = (
                    float(npmi[topic_id][local_id, selected_positions].mean())
                    if selected_positions
                    else 0.0
                )
                score = float(base_score[topic_id, word_id]) + gamma * coherence
                if score > chosen_score or (
                    score == chosen_score and chosen_word is not None and word_id < chosen_word
                ):
                    chosen_word = word_id
                    chosen_score = score

            if chosen_word is None:  # Guard against future changes to pool sizing.
                raise RuntimeError(
                    "candidate pool exhausted during topic-word allocation: "
                    f"topic={topic_id}, selected={len(selected[topic_id])}"
                )

            selected[topic_id].append(chosen_word)
            globally_used.add(chosen_word)

    indices = tuple(tuple(topic_words) for topic_words in selected)
    words = tuple(tuple(vocab[word_id] for word_id in topic_words) for topic_words in indices)
    return DecodedTopicWords(words=words, indices=indices)


def _as_count_matrix(document_term: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    bow = sp.csr_matrix(document_term, dtype=np.float64)
    if bow.ndim != 2:
        raise ValueError("document_term must be two-dimensional")
    if bow.shape[0] == 0 or bow.shape[1] == 0:
        raise ValueError("document_term must contain documents and terms")
    if not np.all(np.isfinite(bow.data)):
        raise ValueError("document_term must contain only finite values")
    if np.any(bow.data < 0.0):
        raise ValueError("document_term must be non-negative")
    return bow


def _validate_matrices(
    document_term: sp.spmatrix | np.ndarray,
    posteriors: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    bow = _as_count_matrix(document_term)
    probabilities = np.asarray(posteriors, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("posteriors must be two-dimensional")
    if probabilities.shape[0] != bow.shape[0]:
        raise ValueError(
            "posteriors rows must match document_term rows: "
            f"{probabilities.shape[0]} != {bow.shape[0]}"
        )
    if probabilities.shape[1] == 0:
        raise ValueError("posteriors must contain at least one topic")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("posteriors must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("posteriors must be non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("every posterior row must sum to one")
    if np.any(probabilities.sum(axis=0) <= 0.0):
        raise ValueError("every topic must have positive posterior mass")
    return bow, probabilities


def _candidate_npmi(
    binary_bow: sp.csc_matrix,
    candidate_indices: np.ndarray,
    document_frequencies: np.ndarray,
) -> np.ndarray:
    document_count = binary_bow.shape[0]
    subset = binary_bow[:, candidate_indices].astype(np.float64)
    cooccurrence = (subset.T @ subset).toarray()
    marginal_probability = document_frequencies[candidate_indices] / document_count
    joint_probability = cooccurrence / document_count

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(
            np.maximum(joint_probability, 1e-12)
            / np.maximum(
                marginal_probability[:, None] * marginal_probability[None, :],
                1e-12,
            )
        )
        result = pmi / -np.log(np.maximum(joint_probability, 1e-12))

    result[joint_probability <= 0.0] = -1.0
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 1.0)
    return result
