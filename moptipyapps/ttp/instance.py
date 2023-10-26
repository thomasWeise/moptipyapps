"""
An instance of the Traveling Tournament Problem (TTP).

The Traveling Tournament Problem (TTP) describes the logistics of a sports
league. In this league, `n` teams compete. In each time slot, each team plays
against one other team. In each game, one team plays at home and one team
plays away with the other team. In each round, every team plays once against
every other team. The league may have multiple :attr:`~moptipyapps.ttp.\
instance.Instance.rounds`. If there are two rounds, then each team plays
against each other team once at home and once abroad. If a team plays at home
(or abroad) several times in a row, this is called a "streak". There are
minimum and maximum streak length constraints defined, for both at home and
abroad. Additionally, if team A plays a team B in one time slot, then the
exact inverse game cannot take place in the next time slot. A minimum
number of games must take place in between for separation. There can also be
a maximum separation length.

David Van Bulck of the Sports Scheduling Research group, part of the
Faculty of Economics and Business Administration at Ghent University, Belgium,
maintains "RobinX: An XML-driven Classification for Round-Robin Sports
Timetabling", a set of benchmark data instances and results of the TTP.
We provide some of these instances as resources here. You can also download
them directly at <https://robinxval.ugent.be/RobinX/travelRepo.php>. Also,
see <https://robinxval.ugent.be/> for more information.

1. David Van Bulck. Minimum Travel Objective Repository. *RobinX: An
   XML-driven Classification for Round-Robin Sports Timetabling.* Faculty of
   Economics and Business Administration at Ghent University, Belgium.
   https://robinxval.ugent.be/
2. Kelly Easton, George L. Nemhauser, and Michael K. Trick. The Traveling
   Tournament Problem Description and Benchmarks. In *Principles and Practice
   of Constraint Programming (CP'01),*  November 26 - December 1, 2001, Paphos,
   Cyprus, pages 580-584, Berlin/Heidelberg, Germany: Springer.
   ISBN: 978-3-540-42863-3. https://doi.org/10.1007/3-540-45578-7_43
   https://www.researchgate.net/publication/220270875
"""

from typing import Callable, Final, Iterable, TextIO, cast

import numpy as np
from defusedxml import ElementTree  # type: ignore
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT, int_range_to_dtype
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, check_to_int_range, type_error

from moptipyapps.tsp.instance import Instance as TSPInstance
from moptipyapps.ttp.robinx import open_resource_stream


def _from_stream(stream: TextIO) -> "Instance":
    """
    Read a TTP instance from an `robinxval.ugent.be`-formatted XML file.

    This procedure ignores most of the data in the file and only focuses on
    the instance name, the team names, and the distance matrix as well as the
    constraints for home streak length, away streak length, and same-game
    separations. Everything else is ignored.

    :param stream: the text stream
    :return: the instance
    """
    used_names: set[str] = set()
    team_names: dict[int, str] = {}
    distances: dict[tuple[int, int], int] = {}
    name: str | None = None
    rounds: int | None = None
    home_streak_min: int | None = None
    home_streak_max: int | None = None
    away_streak_min: int | None = None
    away_streak_max: int | None = None
    separation_min: int | None = None
    separation_max: int | None = None

    for _event, element in ElementTree.iterparse(stream, forbid_dtd=True,
                                                 forbid_entities=True,
                                                 forbid_external=True):
        if element.tag is None:
            continue
        tag: str = element.tag.strip().lower()
        if tag == "instancename":
            if name is None:
                name = sanitize_name(element.text).lower()
                if name in used_names:
                    raise ValueError(
                        f"name {element.text!r} invalid, as it "
                        f"maps to {name!r}, which is already used.")
                used_names.add(name)
            else:
                raise ValueError(f"already got name={name!r}, but tag "
                                 f"'InstanceName' appears again?")
        elif tag == "distance":
            t1: int = check_to_int_range(element.attrib["team1"],
                                         "team1", 0, 1_000_000)
            t2: int = check_to_int_range(element.attrib["team2"],
                                         "team2", 0, 1_000_000)
            dst: int = check_to_int_range(
                element.attrib["dist"], "dist", 0, 1_000_000_000_000)
            if t1 == t2:
                if dst == 0:
                    continue
                raise ValueError(f"distance for team1={t1}, team2={t2} is"
                                 f" {dst} but must be 0.")
            tpl: tuple[int, int] = (t1, t2)
            if tpl in distances:
                raise ValueError(
                    f"got distance={dst} for {tpl!r}, but "
                    f"{distances[tpl]} was already specified before")
            distances[tpl] = dst
        elif element.tag == "team":
            team: int = check_to_int_range(element.attrib["id"], "id",
                                           0, 1_000_000)
            nn: str = element.attrib["name"]
            tname: str = nn.strip()
            if tname in used_names:
                raise ValueError(f"name {nn!r} is invalid, as it maps to "
                                 f"{tname!r}, which is already used.")
            used_names.add(tname)
            team_names[team] = tname
        elif element.tag == "numberroundrobin":
            if rounds is not None:
                raise ValueError(f"rounds already set to {rounds}")
            rounds = check_to_int_range(element.text, "rounds", 1, 1000)
        elif tag == "ca3":
            if "mode1" not in element.attrib:
                continue
            if "mode2" not in element.attrib:
                continue
            if element.attrib["mode2"].lower() != "games":
                continue
            mode = element.attrib["mode1"].lower()
            mi = check_to_int_range(
                element.attrib["min"], "min", 0, 1_000_000) \
                if "min" in element.attrib else None
            ma = check_to_int_range(
                element.attrib["max"], "max", 1, 1_000_000) \
                if "max" in element.attrib else None
            if mode == "h":
                if mi is not None:
                    if home_streak_min is not None:
                        raise ValueError("minimum home streak already defined")
                    home_streak_min = 1 if mi < 1 else mi
                if ma is not None:
                    if home_streak_max is not None:
                        raise ValueError("maximum home streak already defined")
                    home_streak_max = ma
            elif mode == "a":
                if mi is not None:
                    if away_streak_min is not None:
                        raise ValueError("minimum away streak already defined")
                    away_streak_min = 1 if mi < 1 else mi
                if ma is not None:
                    if away_streak_max is not None:
                        raise ValueError("maximum away streak already defined")
                    away_streak_max = ma
        elif tag == "se1":
            mi = check_to_int_range(
                element.attrib["min"], "min", 0, 1_000_000) \
                if "min" in element.attrib else None
            ma = check_to_int_range(
                element.attrib["max"], "max", 0, 1_000_000) \
                if "max" in element.attrib else None
            if mi is not None:
                if separation_min is not None:
                    raise ValueError("minimum separation already defined")
                separation_min = mi
            if ma is not None:
                if separation_max is not None:
                    raise ValueError("maximum separation already defined")
                separation_max = ma

    if name is None:
        raise ValueError("did not find instance name")
    n_teams: Final[int] = len(team_names)
    if n_teams <= 0:
        raise ValueError("did not find any team name")
    if len(used_names) != (n_teams + 1):
        raise ValueError(f"set of used names {used_names!r} has wrong "
                         f"length, should be {n_teams + 1}.")
    dm: np.ndarray = np.zeros((n_teams, n_teams), DEFAULT_INT)
    for tup, dst in distances.items():
        dm[tup[0], tup[1]] = dst

    if rounds is None:
        rounds = 2
    ll: Final[int] = rounds * n_teams - 1
    if home_streak_min is None:
        home_streak_min = 1
    if home_streak_max is None:
        home_streak_max = min(max(home_streak_min, 3), ll)
    if away_streak_min is None:
        away_streak_min = 1
    if away_streak_max is None:
        away_streak_max = min(max(away_streak_min, 3), ll)
    if separation_min is None:
        separation_min = 1
    if separation_max is None:
        separation_max = min(max(separation_min, 1), ll)

    return Instance(name, dm, [team_names[i] for i in range(n_teams)],
                    rounds, home_streak_min, home_streak_max, away_streak_min,
                    away_streak_max, separation_min, separation_max)


#: The instances made available within this package are taken from
#: <https://robinxval.ugent.be/RobinX/travelRepo.php>, where the following
#: descriptions are given:
#: - *Constant Distance (`con*`):* The constant distance instances are the
#:   most simple instances in which the distance between the home venues of
#:   any two teams is one. In this case, Urrutia and Ribeiro showed that
#:   distance minimization is equivalent with break maximization.
#: - *Circular Distance (`circ*`):* Somewhat similar are the circular
#:   distance instances in which the teams' venues are placed on a
#:   circle. Any two consecutive teams are connected by an edge and the
#:   distance between two teams is equal to the minimal number of edges that
#:   must be traversed to get to the other team. Although traveling
#:   salesperson problems with a circular distance matrix have a trivial
#:   solution, it remains challenging to solve circular traveling tournament
#:   instances.
#: - *Galaxy (`gal*`):* This artificial instance class consists of the Galaxy
#:   instances that use a 3D-space that embeds the Earth and 39 other
#:   exoplanets.
#: - *National league (`nl*`):* The `nl`-instances are based on air distance
#:   between the city centers from teams in the National League of the Major
#:   League Baseball.
#: - *National football league (`nfl*`):* The NFL-instances are based on air
#:   distance between the city centers from teams in the National Football
#:   League.
#: - *Super 14 (`sup*`):* The super 14 instances are based on air distance
#:   between the city centers from teams in the Super 14 rugby cup.
#: - *Brazilian (`bra24`)):* The Brazilian instance is based on the air
#:   distance between the home cities of 24 teams in the main division of the
#:   2003 edition of the Brazilian soccer championship.
#: - *Linear (`line*`):* In the linear instances, `n` teams are located on a
#:   straight line with a distance of one unit separating each pair of
#:   adjacent teams.
#: - *Increasing distance (`incr*`):* In the increasing distance instances,
#:   `n` teams are located on a straight line with an increasing distance
#:   separating each pair of adjacent teams such that the distance between
#:   team `k` and `k+1` equals `k`.
_INSTANCES: Final[tuple[str, ...]] = (
    "bra24", "circ4", "circ6", "circ8", "circ10", "circ12", "circ14",
    "circ16", "circ18", "circ20", "circ22", "circ24", "circ26", "circ28",
    "circ30", "circ32", "circ34", "circ36", "circ38", "circ40", "con4",
    "con6", "con8", "con10", "con12", "con14", "con16", "con18", "con20",
    "con22", "con24", "con26", "con28", "con30", "con32", "con34", "con36",
    "con38", "con40", "gal4", "gal6", "gal8", "gal10", "gal12", "gal14",
    "gal16", "gal18", "gal20", "gal22", "gal24", "gal26", "gal28", "gal30",
    "gal32", "gal34", "gal36", "gal38", "gal40", "incr4", "incr6", "incr8",
    "incr10", "incr12", "incr14", "incr16", "incr18", "incr20", "incr22",
    "incr24", "incr26", "incr28", "incr30", "incr32", "incr34", "incr36",
    "incr38", "incr40", "line4", "line6", "line8", "line10", "line12",
    "line14", "line16", "line18", "line20", "line22", "line24", "line26",
    "line28", "line30", "line32", "line34", "line36", "line38", "line40",
    "nfl16", "nfl18", "nfl20", "nfl22", "nfl24", "nfl26", "nfl28", "nfl30",
    "nfl32", "nl4", "nl6", "nl8", "nl10", "nl12", "nl14", "nl16", "sup4",
    "sup6", "sup8", "sup10", "sup12", "sup14")


class Instance(TSPInstance):
    """An instance of Traveling Tournament Problem (TTP)."""

    #: the names of the teams
    teams: tuple[str, ...]
    #: the number of rounds
    rounds: int
    #: the minimum number of games that can be played at home in a row
    home_streak_min: int
    #: the maximum number of games that can be played at home in a row
    home_streak_max: int
    #: the minimum number of games that can be played away in a row
    away_streak_min: int
    #: the maximum number of games that can be played away in a row
    away_streak_max: int
    #: the minimum number of games between a repetition of a game setup
    separation_min: int
    #: the maximum number of games between a repetition of a game setup
    separation_max: int
    #: the data type to be used for plans
    game_plan_dtype: np.dtype

    def __new__(cls, name: str, matrix: np.ndarray, teams: Iterable[str],
                rounds: int, home_streak_min: int, home_streak_max: int,
                away_streak_min: int, away_streak_max: int,
                separation_min: int, separation_max: int,
                tour_length_lower_bound: int = 0) -> "Instance":
        """
        Create an instance of the Traveling Salesperson Problem.

        :param cls: the class
        :param name: the name of the instance
        :param matrix: the matrix with the data (will be copied)
        :param teams: the iterable with the team names
        :param rounds: the number of rounds
        :param tour_length_lower_bound: the lower bound of the tour length
        :param home_streak_min: the minimum number of games that can be played
            at home in a row
        :param home_streak_max: the maximum number of games that can be played
            at home in a row
        :param away_streak_min: the minimum number of games that can be played
            away in a row
        :param away_streak_max: the maximum number of games that can be played
            away in a row
        :param separation_min: the minimum number of games between a repetition
            of a game setup
        :param separation_max: the maximum number of games between a repetition
            of a game setup
        """
        names: Final[tuple[str, ...]] = tuple(map(str.strip, teams))
        n: Final[int] = len(names)
        if (n % 2) != 0:
            raise ValueError(f"the number of teams must be even, but is {n}.")
        if n != len(set(names)):
            raise ValueError(f"some team name appears twice in {teams!r} "
                             f"after fixing it to {names!r}.")
        for nn in names:
            for char in nn:
                if char.isspace():
                    raise ValueError(
                        f"team name must not contain space, but {nn} does.")

        obj: Final[Instance] = cast(Instance, super().__new__(
            cls, name, tour_length_lower_bound, matrix, rounds * n))

        if (obj.shape[0] != n) or (obj.shape[1] != n) or (obj.n_cities != n):
            raise ValueError(f"inconsistent n_teams={n}, n_cities="
                             f"{obj.n_cities} and shape={obj.shape}")
        #: the names of the teams that compete
        obj.teams = names
        #: the number of rounds
        obj.rounds = check_int_range(rounds, "rounds", 1, 100)
        ll: Final[int] = rounds * n - 1  # an upper bound for streaks
        #: the minimum number of games that can be played at home in a row
        obj.home_streak_min = check_int_range(
            home_streak_min, "home_streak_min", 1, ll)
        #: the maximum number of games that can be played at home in a row
        obj.home_streak_max = check_int_range(
            home_streak_max, "home_streak_max", home_streak_min, ll)
        #: the minimum number of games that can be played away in a row
        obj.away_streak_min = check_int_range(
            away_streak_min, "away_streak_min", 1, ll)
        #: the maximum number of games that can be played away in a row
        obj.away_streak_max = check_int_range(
            away_streak_max, "away_streak_max", away_streak_min, ll)
        #: the minimum number of games between a repetition of a game setup
        obj.separation_min = check_int_range(
            separation_min, "separation_min", 0, ll)
        #: the maximum number of games between a repetition of a game setup
        obj.separation_max = check_int_range(
            separation_max, "separation_max", separation_min, ll)
        #: the data type to be used for the game plans
        obj.game_plan_dtype = int_range_to_dtype(-n, n)
        return obj

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the instance to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("I") as kv:
        ...         Instance.from_resource("gal4").log_parameters_to(kv)
        ...     print(repr('@'.join(l.get_log())))
        'BEGIN_I@name: gal4@class: moptipyapps.ttp.instance.Instance@\
nCities: 4@tourLengthLowerBound: 67@tourLengthUpperBound: 160@symmetric: T@\
dtype: h@rounds: 2@homeStreakMin: 1@homeStreakMax: 3@\
awayStreakMin: 1@awayStreakMax: 3@separationMin: 1@separationMax: 6@\
gamePlanDtype: b@END_I'
        """
        super().log_parameters_to(logger)
        logger.key_value("rounds", self.rounds)
        logger.key_value("homeStreakMin", self.home_streak_min)
        logger.key_value("homeStreakMax", self.home_streak_max)
        logger.key_value("awayStreakMin", self.away_streak_min)
        logger.key_value("awayStreakMax", self.away_streak_max)
        logger.key_value("separationMin", self.separation_min)
        logger.key_value("separationMax", self.separation_max)
        logger.key_value("gamePlanDtype", self.game_plan_dtype.char)

    @staticmethod
    def from_file(path: str, lower_bound_getter: Callable[[
            str], int] | None = None) -> "Instance":
        """
        Read a TTP instance from a `robinX` formatted XML file.

        :param path: the path to the file
        :param lower_bound_getter: ignored
        :return: the instance

        >>> from os.path import dirname
        >>> inst = Instance.from_file(dirname(__file__) + "/robinx/con20.xml")
        >>> inst.name
        'con20'
        """
        file: Final[Path] = Path.file(path)
        with file.open_for_read() as stream:
            try:
                return _from_stream(cast(TextIO, stream))
            except (TypeError, ValueError) as err:
                raise ValueError(f"error when parsing file {file!r}") from err

    @staticmethod
    def list_resources(symmetric: bool = True,
                       asymmetric: bool = True) -> tuple[str, ...]:
        """
        Get a tuple of all the TTP instances available as resource.

        All instances of the `robinX` set provided here are symmetric.

        :param symmetric: include the instances with symmetric distance
            matrices
        :param asymmetric: include the asymmetric instances with asymmetric
            distance matrices
        :return: the tuple with the instance names

        >>> len(Instance.list_resources())
        118
        >>> len(Instance.list_resources(False, True))
        0
        >>> len(Instance.list_resources(True, False))
        118
        """
        return _INSTANCES if symmetric else ()

    @staticmethod
    def from_resource(name: str) -> "Instance":
        """
        Load a TTP instance from a resource.

        :param name: the name string
        :return: the instance

        >>> insta = Instance.from_resource("bra24")
        >>> insta.n_cities
        24
        >>> insta.name
        'bra24'
        >>> insta.teams[0]
        'Atl.Mineiro'
        >>> insta.teams[1]
        'Atl.Paranaense'
        >>> insta.rounds
        2
        >>> insta.home_streak_min
        1
        >>> insta.home_streak_max
        3
        >>> insta.away_streak_min
        1
        >>> insta.away_streak_max
        3
        >>> insta.separation_min
        1
        >>> insta.separation_max
        46
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        container: Final = Instance.from_resource
        inst_attr: Final[str] = f"__inst_{name}"
        if hasattr(container, inst_attr):  # instance loaded?
            return cast(Instance, getattr(container, inst_attr))

        with open_resource_stream(f"{name}.xml") as stream:
            inst: Final[Instance] = _from_stream(stream)

        if inst.name != name:
            raise ValueError(f"got {inst.name!r} for instance {name!r}?")
        if inst.n_cities <= 1000:
            setattr(container, inst_attr, inst)
        return inst
