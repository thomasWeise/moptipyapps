"""Run small experiments applying Algorithms Specialized to the TSP."""

from moptipy.api.execution import Execution
from moptipy.spaces.permutations import Permutations

from moptipyapps.tsp.ea1p1_revn import TSPEA1p1revn
from moptipyapps.tsp.fea1p1_revn import TSPFEA1p1revn
from moptipyapps.tsp.instance import Instance
from moptipyapps.tsp.tour_length import TourLength

# load the problem instance and define search space
instance = Instance.from_resource("burma14")  # pick instance burma14
space = Permutations.standard(instance.n_cities)

# the specialized algorithms that we will try out
algorithms = [TSPEA1p1revn, TSPFEA1p1revn]

# Apply the algorithms to the smallest instance, burma14.
for constructor in algorithms:
    algorithm = constructor(instance)
    with Execution()\
            .set_rand_seed(1)\
            .set_solution_space(space)\
            .set_algorithm(algorithm)\
            .set_objective(TourLength(instance))\
            .set_max_fes(1000)\
            .execute() as process:
        print(f"{algorithm}: {process.get_best_f()}")
