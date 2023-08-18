"""Test the tour length objective function on the TSP."""

from moptipyapps.tests.on_tsp import validate_objective_on_tsp
from moptipyapps.tsp.tour_length import TourLength


def test_tour_length_on_tsp() -> None:
    """Test the tour length objective function on the TSP."""
    validate_objective_on_tsp(TourLength)
