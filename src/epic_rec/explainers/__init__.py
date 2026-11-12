"""Exact and approximate explainers with a common, verified output (:class:`Explanation`)."""

from epic_rec.explainers.api import EXACT_STATUSES, Explanation, explain
from epic_rec.explainers.beam import beam_search
from epic_rec.explainers.common import static_order, violation
from epic_rec.explainers.continuous import continuous_mask
from epic_rec.explainers.greedy import CRITERIA, greedy
from epic_rec.explainers.repair import repair

__all__ = ["CRITERIA", "EXACT_STATUSES", "Explanation", "beam_search", "continuous_mask", "explain", "greedy",
           "repair", "static_order", "violation"]
