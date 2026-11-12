import numpy as np
import pytest

from epic_rec.data.synthetic import make_synthetic
from epic_rec.models.path_additive import PathAdditiveRecommender
from epic_rec.optimization.instance import ExplanationInstance


@pytest.fixture(scope="session")
def ds():
    return make_synthetic(seed=0)


@pytest.fixture(scope="session")
def model(ds):
    return PathAdditiveRecommender.fit(ds, {"seed": 0, "bpr_samples": 20000})


@pytest.fixture(scope="session")
def instances(model):
    """(user, target) model instances with a small mask domain, so enumeration is exact."""
    out = []
    for u in range(model.n_users):
        inst = ExplanationInstance(model, u, K=5)
        for t in inst.L[:3]:
            inst.set_target(int(t))
            if 0 < len(inst.scope) <= 10:
                out.append((u, int(t)))
    return out[:25]


def random_instance(rng: np.random.Generator) -> ExplanationInstance | None:
    h, n = int(rng.integers(2, 9)), int(rng.integers(6, 16))
    K = int(rng.integers(2, min(5, n - 2)))
    A = rng.random((h, n)) * (rng.random((h, n)) < 0.5)
    A[:, rng.random(n) < 0.2] = 0  # items whose score cannot change
    s = np.round(rng.random(n) * 4 + A.sum(0) * rng.random(), 1)  # rounding creates exact ties
    inst = ExplanationInstance.from_arrays(A, A, s, K, alpha=float(rng.choice([0.25, 0.5, 1.0])))
    inst.set_target(int(inst.L[rng.integers(0, K)]))
    if len(inst.scope) == 0 or len(inst.outsiders) < 2:
        return None
    return inst
