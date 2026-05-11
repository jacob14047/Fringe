"""
nuqkd.rag.retriever
=====================
Retrieval engine for the quantum attack knowledge base.

Two retrieval modes:

TF-IDF (default, no extra dependencies):
    Fast keyword-based similarity using scikit-learn's TfidfVectorizer.
    Suitable for structured queries like "detector efficiency mismatch attack".

Semantic (optional, requires sentence-transformers):
    Dense vector retrieval using a local sentence embedding model.
    Better for natural-language queries from the LLM agents.

Hybrid scoring:
    final_score = alpha * tfidf_score + (1-alpha) * signature_match_score

where signature_match_score rewards documents whose "signatures" list
contains terms that match observations in the BeliefState.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nuqkd.rag.knowledge_base import KBDocument, get_all_documents
from nuqkd.belief.state import BeliefState

logger = logging.getLogger(__name__)


class QKDRetriever:
    """
    Retrieval engine over the quantum attack knowledge base.

    Parameters
    ----------
    alpha : float
        Weight of TF-IDF score vs. signature match score.
    top_k : int
        Default number of documents to return.
    use_semantic : bool
        If True, attempt to load sentence-transformers for dense retrieval.
    semantic_model : str
        HuggingFace model ID for embeddings (runs locally).
    """

    def __init__(self,
                 alpha: float = 0.6,
                 top_k: int = 5,
                 use_semantic: bool = False,
                 semantic_model: str = "all-MiniLM-L6-v2") -> None:
        self.alpha      = alpha
        self.top_k      = top_k
        self.documents  = get_all_documents()

        # Build TF-IDF index
        corpus = [f"{d.title} {d.content} {' '.join(d.tags)}"
                  for d in self.documents]
        self._tfidf     = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)

        # Optional dense encoder
        self._encoder = None
        self._dense_matrix = None
        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(semantic_model)
                texts = [f"{d.title}. {d.content[:300]}" for d in self.documents]
                self._dense_matrix = self._encoder.encode(texts,
                                                           convert_to_numpy=True,
                                                           show_progress_bar=False)
                logger.info("Semantic encoder loaded: %s", semantic_model)
            except ImportError:
                logger.warning("sentence-transformers not installed; "
                               "falling back to TF-IDF only.")

    # ------------------------------------------------------------------
    # Primary retrieval
    # ------------------------------------------------------------------

    def retrieve(self,
                 query: str,
                 belief: Optional[BeliefState] = None,
                 top_k: Optional[int] = None,
                 filter_tags: Optional[List[str]] = None) -> List[Tuple[KBDocument, float]]:
        """
        Retrieve relevant attack documents for a given query.

        Parameters
        ----------
        query : str
            Natural-language query (from the RAG Querier agent).
        belief : BeliefState | None
            Current belief state — used for signature matching.
        top_k : int | None
            Override default top-k.
        filter_tags : list[str] | None
            Only return documents with at least one of these tags.

        Returns
        -------
        list of (KBDocument, score) sorted by score descending.
        """
        k       = top_k or self.top_k
        indices = list(range(len(self.documents)))

        # Tag filter
        if filter_tags:
            indices = [
                i for i in indices
                if any(t in self.documents[i].tags for t in filter_tags)
            ]
            if not indices:
                indices = list(range(len(self.documents)))

        # TF-IDF scores
        tfidf_scores = self._tfidf_score(query, indices)

        # Semantic scores (if available)
        if self._encoder is not None and self._dense_matrix is not None:
            sem_scores = self._semantic_score(query, indices)
            base_scores = self.alpha * tfidf_scores + (1.0 - self.alpha) * sem_scores
        else:
            base_scores = tfidf_scores

        # Signature matching bonus (from belief state)
        if belief is not None:
            sig_scores = self._signature_score(belief, indices)
            final_scores = 0.7 * base_scores + 0.3 * sig_scores
        else:
            final_scores = base_scores

        # Rank and return
        ranked  = sorted(zip(indices, final_scores), key=lambda x: -x[1])[:k]
        results = [(self.documents[i], float(s)) for i, s in ranked if s > 0.0]
        return results

    def retrieve_for_belief(self,
                             belief: BeliefState,
                             top_k: Optional[int] = None) -> List[Tuple[KBDocument, float]]:
        """
        Retrieve attack documents most consistent with the current belief state.

        Constructs the query automatically from the top anomaly scores and
        parameter estimates in the belief state.
        """
        query_parts = []

        # Anomalies
        for vuln_id, score in belief.top_anomalies(threshold=0.2).items():
            label = {
                "TC-01": "timing side channel basis dependent",
                "DM-01": "detector efficiency mismatch time shift",
                "MU-01": "mean photon number high PNS attack",
                "RNG-01": "biased random number generator basis prediction",
                "DT-01": "decoy pulse timing correlation identification",
                "DB-01": "detector blinding linear mode",
                "PA-01": "privacy amplification weak seed entropy",
            }.get(vuln_id, vuln_id)
            query_parts.append(label)

        # Parameter anomalies
        mu_est = belief.get_parameter("mu_eff")
        if mu_est and mu_est.mean > 0.5 and mu_est.n_obs > 0:
            query_parts.append("high mean photon number weak coherent pulse")

        eta_d = belief.get_parameter("eta_delta")
        if eta_d and abs(eta_d.mean) > 0.03:
            query_parts.append("detector efficiency mismatch")

        if not query_parts:
            query_parts = ["quantum channel vulnerability attack"]

        query = " ".join(query_parts)
        return self.retrieve(query, belief=belief, top_k=top_k)

    # ------------------------------------------------------------------
    # Format for LLM context
    # ------------------------------------------------------------------

    def format_for_llm(self,
                        results: List[Tuple[KBDocument, float]],
                        max_chars: int = 4000) -> str:
        """
        Format retrieved documents as a compact string for LLM injection.
        """
        lines = ["=== RETRIEVED ATTACK KNOWLEDGE ===\n"]
        chars_used = 0

        for doc, score in results:
            header = f"[{doc.doc_id}] {doc.title} (relevance={score:.2f})"
            prereqs = "Prerequisites: " + str(doc.prerequisites)
            sigs    = "Signatures: " + "; ".join(doc.signatures[:3])
            excerpt = doc.content[:400] + "..."

            block = f"{header}\n{prereqs}\n{sigs}\n{excerpt}\n"
            if chars_used + len(block) > max_chars:
                break
            lines.append(block)
            chars_used += len(block)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _tfidf_score(self, query: str,
                      indices: List[int]) -> np.ndarray:
        q_vec    = self._tfidf.transform([query])
        sims     = cosine_similarity(q_vec, self._tfidf_matrix[indices]).flatten()
        return sims

    def _semantic_score(self, query: str,
                          indices: List[int]) -> np.ndarray:
        q_emb = self._encoder.encode([query], convert_to_numpy=True)
        doc_emb = self._dense_matrix[indices]
        sims = cosine_similarity(q_emb, doc_emb).flatten()
        return (sims + 1.0) / 2.0   # normalise to [0,1]

    def _signature_score(self,
                          belief: BeliefState,
                          indices: List[int]) -> np.ndarray:
        """
        Score documents based on how many of their signatures match
        observations in the belief state.
        """
        # Build a flat set of observation strings from the belief state
        obs_tokens: set = set()
        for rec in belief.observation_log[-20:]:   # last 20 observations
            for k, v in rec.get("data", {}).items():
                obs_tokens.add(k.lower())
                if isinstance(v, bool) and v:
                    obs_tokens.add(k.lower() + "=true")
                elif isinstance(v, (int, float)):
                    obs_tokens.add(k.lower())

        # Anomaly IDs
        for vid, score in belief.top_anomalies(0.1).items():
            obs_tokens.add(vid.lower())

        scores = np.zeros(len(indices))
        for j, i in enumerate(indices):
            doc = self.documents[i]
            matches = sum(
                1 for sig in doc.signatures
                if any(tok in sig.lower() for tok in obs_tokens)
            )
            scores[j] = matches / max(len(doc.signatures), 1)

        return scores
