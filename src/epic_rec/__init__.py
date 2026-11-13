"""EPIC-Rec: evidence-preserving counterfactual explanations for top-K graph recommendation.

Main entry points:

* :class:`epic_rec.models.PathAdditiveRecommender` - the path-additive recommender whose scores decompose
  into individual path contributions;
* :class:`epic_rec.optimization.ExplanationInstance` - one (user, target) problem;
* :func:`epic_rec.explainers.explain` - exact or approximate explanation with independent verification;
* :mod:`epic_rec.experiments` - configurable, resumable experiment runner.
"""

__version__ = "0.1.0"
