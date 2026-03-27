import numpy as np

from project.core.fitting.fitting_strategy import FittingController, StrategySelector


class _DummyProvider:
    def get_fitting_points(self, eye):
        return []

    def get_constraint_points(self, eye):
        return []

    def get_data_quality(self, eye):
        return {"point_count": 0, "visibility": 0.0}


def test_fit_circle_to_points_returns_true_circumcenter():
    controller = FittingController(data_provider=_DummyProvider(), strategy_selector=StrategySelector([]))
    p1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    p3 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    center, radius = controller._fit_circle_to_points([p1, p2, p3])

    assert np.allclose(center, np.array([0.0, 0.0, 0.0]), atol=1e-6)
    assert abs(radius - 1.0) < 1e-6
