"""Test the (1+1) EA with sub-tour reversal on the TSP."""

from moptipyapps.tests.on_tsp import validate_algorithm_on_tsp
from moptipyapps.tsp.ea1p1_revn import TSPEA1p1revn


def test_ea1p1_revn_on_tsp() -> None:
    """Test the (1+1) EA with sub-tour reversal on the TSP."""
    validate_algorithm_on_tsp(lambda i, _: TSPEA1p1revn(i),
                              asymmetric=False)
