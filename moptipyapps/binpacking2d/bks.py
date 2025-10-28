"""In this file, we provide a set of best-known solutions for the 2D-BPP."""

from dataclasses import dataclass
from importlib import resources  # nosem
from math import inf, isfinite
from re import Pattern
from re import compile as p_compile
from statistics import mean
from typing import Any, Callable, Final, Iterable, Mapping, cast

from pycommons.ds.immutable_map import immutable_mapping
from pycommons.io.path import UTF8
from pycommons.math.int_math import try_int, try_int_mul
from pycommons.strings.chars import WHITESPACE
from pycommons.strings.string_tools import replace_regex
from pycommons.types import check_int_range

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing_result import (
    BIN_COUNT_NAME,
    PackingResult,
)


def __load_references() -> Mapping[str, str]:
    """
    Load the references from the BibTeX resource.

    This is not a fully-fledged BibTeX parser. It is a very simple and crude
    method to extract bibliography keys and data.

    :return: the immutable reference dictionary, with keys sorted
        alphabetically
    """
    current_lines: list[str] = []
    key: str | None = None
    mode: int = 0
    found: Final[dict[str, str]] = {}
    single_ws: Final[Pattern] = p_compile(f"[{WHITESPACE}]")
    multi_ws: Final[Pattern] = p_compile(f"[{WHITESPACE}][{WHITESPACE}]+")
    with resources.files(__package__).joinpath(
            "bks.bib").open("r", encoding=UTF8) as stream:
        for rline in stream:
            line: str = str.rstrip(rline)
            if str.__len__(line) <= 0:
                continue
            if mode == 0:
                line = replace_regex(single_ws, "", line)
                if str.startswith(line, "@"):
                    mode = 1
                    start_idx: int = line.index("{")
                    end_idx: int = line.index(",", start_idx + 1)
                    key = str.strip(line[start_idx + 1:end_idx])
                    if str.__len__(key) <= 0:
                        raise ValueError(f"Invalid key {key!r} in {line!r}.")
                    current_lines.append(line)
                    if key in found:
                        raise ValueError(f"Duplicate key {key!r}.")
            elif mode == 1:
                if str.strip(line) in {"}", "},"}:
                    mode = 0
                    current_lines.append("}")
                    found[key] = str.strip("\n".join(current_lines))
                    current_lines.clear()
                else:
                    current_lines.append(replace_regex(multi_ws, " ", line))
    if dict.__len__(found) <= 0:
        raise ValueError("Found no references!")
    return immutable_mapping({k: found[k] for k in sorted(found.keys())})


#: the references to literature
_REFERENCES: Final[Mapping[str, str]] = __load_references()

#: a constant for no reference
NO_REF: Final[str] = "{{NO_REF}}"


@dataclass(frozen=True, init=False, order=True, eq=True)
class Element:
    """The algorithm or instance group specification."""

    #: the name or name prefix
    name: str
    #: the name suffix, if any, and the empty string otherwise
    name_suffix: str
    #: the reference to the related work
    reference: str

    def __init__(self, name: str,
                 name_suffix: str, reference: str = NO_REF) -> None:
        """
        Initialize this element.

        :param name: the name or name prefix
        :param name_suffix: the name suffix, or an empty string
        :param reference: the reference
        """
        name = str.strip(name)
        if str.__len__(name) <= 0:
            raise ValueError(f"Invalid name {name!r}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "name_suffix", str.strip(name_suffix))
        reference = str.strip(reference)
        for ref in reference.split(","):
            if ref == NO_REF:
                continue
            if ref not in _REFERENCES:
                raise ValueError(f"Invalid reference {ref!r}.")
        object.__setattr__(self, "reference", reference)

    def get_bibtex(self) -> Iterable[str]:
        """
        Get the BibTeX for this element.

        :return: the BibTeX string
        """
        return (_REFERENCES[r] for r in self.reference.split(","))


#: the BRKGA for 2D bin packing with rotation
BRKGA_BPP_2R: Final[Element] = Element(
    "BRKGA", "BRKGABPPRTR", "GR2013ABRKGAF2A3BPP")
#: the BRKGA for 2D bin packing without rotation
BRKGA_BPP_ANB: Final[Element] = Element(
    "BRKGA", "BRKGABPPRANB", "GR2013ABRKGAF2A3BPP")
#: the grasp/vnd for 2D bin packing without rotation
GRASP_VND: Final[Element] = Element(
    "GRASPVNDGRASP", "GRASPVNDVND", "PAVOT2010AHGVAFTATDBP")
#: the price-and-cut algorithm for 2D bin packing with rotation
PAC: Final[Element] = Element("PAC", "", "CGRS2020PACATSMTOOSFT2BPP")
#: the HHANO-R algorithm for 2D bin packing with rotation
HHANO_R: Final[Element] = Element("HHANO", "HHANOR", "BDC2015RHHAFTOONO2BPP")
#: the HHANO-SR algorithm for 2D bin packing with rotation
HHANO_SR: Final[Element] = Element("HHANO", "HHANOSR", "BDC2015RHHAFTOONO2BPP")
#: the MXGA algorithm for 2D bin packing with rotation
MXGA: Final[Element] = Element("MXGA", "", "L2008AGAFTDBPP")
#: the LGFi algorithm
EALGFI: Final[Element] = Element(
    "EALGFIEA", "EALGFILGFI", "BS2013ST2BPPBMOAHEA")

#: the small A instances
A_SMALL: Final[Element] = Element("a", "small", "MAVdC2010AFMFTTDGCSP")
#: the median sized A instances
A_MED: Final[Element] = Element("a", "med", "MAVdC2010AFMFTTDGCSP")
#: the large A instances
A_LARGE: Final[Element] = Element("a", "large", "MAVdC2010AFMFTTDGCSP")
#: the asqas
ASQAS: Final[Element] = Element("asqas", "", "vdBBMSB2016ASIASSTFI")
#: the first part of the Beng instances
BENG_1_8: Final[Element] = Element("beng", "1-8", "B1982PRPAHA")
#: the second part of the Beng instances
BENG_9_10: Final[Element] = Element("beng", "9-10", "B1982PRPAHA")
#: the class 1-20 benchmarks
CLASS_1_20: Final[Element] = Element(
    "class 1", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-40 benchmarks
CLASS_1_40: Final[Element] = Element(
    "class 1", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-60 benchmarks
CLASS_1_60: Final[Element] = Element(
    "class 1", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-80 benchmarks
CLASS_1_80: Final[Element] = Element(
    "class 1", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-100 benchmarks
CLASS_1_100: Final[Element] = Element(
    "class 1", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 2-20 benchmarks
CLASS_2_20: Final[Element] = Element(
    "class 2", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 2-40 benchmarks
CLASS_2_40: Final[Element] = Element(
    "class 2", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 2-60 benchmarks
CLASS_2_60: Final[Element] = Element(
    "class 2", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 2-80 benchmarks
CLASS_2_80: Final[Element] = Element(
    "class 2", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-100 benchmarks
CLASS_2_100: Final[Element] = Element(
    "class 2", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 3-20 benchmarks
CLASS_3_20: Final[Element] = Element(
    "class 3", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 1-40 benchmarks
CLASS_3_40: Final[Element] = Element(
    "class 3", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 3-60 benchmarks
CLASS_3_60: Final[Element] = Element(
    "class 3", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 3-80 benchmarks
CLASS_3_80: Final[Element] = Element(
    "class 3", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 3-100 benchmarks
CLASS_3_100: Final[Element] = Element(
    "class 3", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 4-20 benchmarks
CLASS_4_20: Final[Element] = Element(
    "class 4", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 4-40 benchmarks
CLASS_4_40: Final[Element] = Element(
    "class 4", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 4-60 benchmarks
CLASS_4_60: Final[Element] = Element(
    "class 4", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 4-80 benchmarks
CLASS_4_80: Final[Element] = Element(
    "class 4", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 4-100 benchmarks
CLASS_4_100: Final[Element] = Element(
    "class 4", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 5-20 benchmarks
CLASS_5_20: Final[Element] = Element(
    "class 5", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 5-40 benchmarks
CLASS_5_40: Final[Element] = Element(
    "class 5", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 5-60 benchmarks
CLASS_5_60: Final[Element] = Element(
    "class 5", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 5-80 benchmarks
CLASS_5_80: Final[Element] = Element(
    "class 5", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 5-100 benchmarks
CLASS_5_100: Final[Element] = Element(
    "class 5", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 6-20 benchmarks
CLASS_6_20: Final[Element] = Element(
    "class 6", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 6-40 benchmarks
CLASS_6_40: Final[Element] = Element(
    "class 6", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 6-60 benchmarks
CLASS_6_60: Final[Element] = Element(
    "class 6", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 6-80 benchmarks
CLASS_6_80: Final[Element] = Element(
    "class 6", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 6-100 benchmarks
CLASS_6_100: Final[Element] = Element(
    "class 6", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 7-20 benchmarks
CLASS_7_20: Final[Element] = Element(
    "class 7", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 7-40 benchmarks
CLASS_7_40: Final[Element] = Element(
    "class 7", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 7-60 benchmarks
CLASS_7_60: Final[Element] = Element(
    "class 7", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 7-80 benchmarks
CLASS_7_80: Final[Element] = Element(
    "class 7", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 7-100 benchmarks
CLASS_7_100: Final[Element] = Element(
    "class 7", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 8-20 benchmarks
CLASS_8_20: Final[Element] = Element(
    "class 8", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 8-40 benchmarks
CLASS_8_40: Final[Element] = Element(
    "class 8", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 8-60 benchmarks
CLASS_8_60: Final[Element] = Element(
    "class 8", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 8-80 benchmarks
CLASS_8_80: Final[Element] = Element(
    "class 8", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 8-100 benchmarks
CLASS_8_100: Final[Element] = Element(
    "class 8", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 9-20 benchmarks
CLASS_9_20: Final[Element] = Element(
    "class 9", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 9-40 benchmarks
CLASS_9_40: Final[Element] = Element(
    "class 9", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 9-60 benchmarks
CLASS_9_60: Final[Element] = Element(
    "class 9", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 9-80 benchmarks
CLASS_9_80: Final[Element] = Element(
    "class 9", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 9-100 benchmarks
CLASS_9_100: Final[Element] = Element(
    "class 9", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")

#: the class 10-20 benchmarks
CLASS_10_20: Final[Element] = Element(
    "class 10", "20", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 10-40 benchmarks
CLASS_10_40: Final[Element] = Element(
    "class 10", "40", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 10-60 benchmarks
CLASS_10_60: Final[Element] = Element(
    "class 10", "60", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 10-80 benchmarks
CLASS_10_80: Final[Element] = Element(
    "class 10", "80", "BW1987TDFBPA,MV1998ESOTTDFBPP")
#: the class 6-100 benchmarks
CLASS_10_100: Final[Element] = Element(
    "class 10", "100", "BW1987TDFBPA,MV1998ESOTTDFBPP")


def __sort_key(txt: str) -> list[str]:
    """
    Get a sort key from a string.

    :param txt: the string
    :return: the sort key
    """
    kys: list[str] = []
    for v in txt.split(","):
        for vv in v.split("-"):
            for vvv in vv.split("_"):
                for vvvv in vvv.split(" "):
                    if vvvv == "small":
                        kys.append("1_small")
                    elif vvvv == "med":
                        kys.append("2_med")
                    elif vvvv == "large":
                        kys.append("3_large")
                    elif vvvv.startswith("asqas"):
                        kys.append(f"zzzzzzzzzzzzzzzzzz{vvvv}")
                    else:
                        kys.append(vvvv.rjust(10, "0")
                                   if vvvv.isdigit() else vvvv)
    return kys


#: The set of class and beng instance groups
CLASS_AND_BENG: Final[tuple[Element, ...]] = (
    BENG_1_8, BENG_9_10, CLASS_1_20, CLASS_1_40, CLASS_1_60, CLASS_1_80,
    CLASS_1_100, CLASS_2_20, CLASS_2_40, CLASS_2_60, CLASS_2_80,
    CLASS_2_100, CLASS_3_20, CLASS_3_40, CLASS_3_60, CLASS_3_80,
    CLASS_3_100, CLASS_4_20, CLASS_4_40, CLASS_4_60, CLASS_4_80,
    CLASS_4_100, CLASS_5_20, CLASS_5_40, CLASS_5_60, CLASS_5_80,
    CLASS_5_100, CLASS_6_20, CLASS_6_40, CLASS_6_60, CLASS_6_80,
    CLASS_6_100, CLASS_7_20, CLASS_7_40, CLASS_7_60, CLASS_7_80,
    CLASS_7_100, CLASS_8_20, CLASS_8_40, CLASS_8_60, CLASS_8_80,
    CLASS_8_100, CLASS_9_20, CLASS_9_40, CLASS_9_60, CLASS_9_80,
    CLASS_9_100, CLASS_10_20, CLASS_10_40, CLASS_10_60, CLASS_10_80,
    CLASS_10_100,
)


def __make_instance_map() -> Mapping[Element, tuple[str, ...]]:
    """
    Make an instance map.

    :return: the instance map.
    """
    groups: Final[list[Element]] = [
        A_SMALL, A_MED, A_LARGE, ASQAS,
        BENG_1_8, BENG_9_10, CLASS_1_20, CLASS_1_40, CLASS_1_60, CLASS_1_80,
        CLASS_1_100, CLASS_2_20, CLASS_2_40, CLASS_2_60, CLASS_2_80,
        CLASS_2_100, CLASS_3_20, CLASS_3_40, CLASS_3_60, CLASS_3_80,
        CLASS_3_100, CLASS_4_20, CLASS_4_40, CLASS_4_60, CLASS_4_80,
        CLASS_4_100, CLASS_5_20, CLASS_5_40, CLASS_5_60, CLASS_5_80,
        CLASS_5_100, CLASS_6_20, CLASS_6_40, CLASS_6_60, CLASS_6_80,
        CLASS_6_100, CLASS_7_20, CLASS_7_40, CLASS_7_60, CLASS_7_80,
        CLASS_7_100, CLASS_8_20, CLASS_8_40, CLASS_8_60, CLASS_8_80,
        CLASS_8_100, CLASS_9_20, CLASS_9_40, CLASS_9_60, CLASS_9_80,
        CLASS_9_100, CLASS_10_20, CLASS_10_40, CLASS_10_60, CLASS_10_80,
        CLASS_10_100]

    data: dict[Element, tuple[str, ...]] = {}
    for group in Instance.list_resources_groups():
        prefix: str = group[0]
        suffix: str = "" if group[1] is None else group[1]
        insts: tuple[str, ...] = group[2]
        found: bool = False
        for ggg in groups:
            if (ggg.name == prefix) and (ggg.name_suffix == suffix):
                if ggg in data:
                    raise ValueError(f"Encountered {ggg} twice?")
                data[ggg] = insts
                found = True
                break
        if not found:
            raise ValueError(f"Did not find {group!r}.")

    return immutable_mapping({k: data[k] for k in sorted(
        data.keys(), key=lambda e: __sort_key(f"{e.name} {e.name_suffix}"))})


#: a mapping of instance groups to instances
GROUPS_TO_INSTANCES: Final[Mapping[Element, tuple[
    str, ...]]] = __make_instance_map()

#: A sort key function for instance groups
INST_GROUP_SORT_KEY: Final[Callable[[Element], int]] \
    = list(GROUPS_TO_INSTANCES.keys()).index


def __make_instance_to_groups() -> Mapping[str, Element]:
    """
    Make a mapping of instances to groups.

    :return: the mapping of instances to groups.
    """
    data: list[tuple[str, Element]] = [(
        v, k) for k, vv in GROUPS_TO_INSTANCES.items() for v in vv]
    data.sort(key=lambda x: __sort_key(x[0]))
    return immutable_mapping(dict(data))


#: a mapping of instances to instance groups
INSTANCES_TO_GROUPS: Final[Mapping[str, Element]] = \
    __make_instance_to_groups()


def __lb_avg_denormalize(
        with_rotation: bool, algo: Element, group: Element,
        value: "int | float", min_result: int = -1_000_000) -> tuple[
        bool, Element, Element, int]:
    """
    De-normalize using the maximum of the Dell'Amico and geometric bound.

    :param with_rotation: is this a result with or without rotation?
    :param algo: the algorithm used
    :param group: the instance group
    :param value: the value
    :param min_result: the minimum denormalized result
    :return: de-normalized average
    """
    return (with_rotation, algo, group, max(
        min_result, round(try_int_mul(sum(Instance.from_resource(
            ins).lower_bound_bins for ins in GROUPS_TO_INSTANCES[group]),
            value))))


#: the related works with rotation averaged
__RW__AVERAGE: Final[tuple[tuple[
    bool, Element, Element, int], ...]] = (
    (True, BRKGA_BPP_2R, CLASS_1_20, 66),
    (False, BRKGA_BPP_ANB, CLASS_1_20, 71),
    (True, BRKGA_BPP_2R, CLASS_1_40, 128),
    (False, BRKGA_BPP_ANB, CLASS_1_40, 134),
    (True, BRKGA_BPP_2R, CLASS_1_60, 195),
    (False, BRKGA_BPP_ANB, CLASS_1_60, 200),
    (True, BRKGA_BPP_2R, CLASS_1_80, 270),
    (False, BRKGA_BPP_ANB, CLASS_1_80, 275),
    (True, BRKGA_BPP_2R, CLASS_1_100, 313),
    (False, BRKGA_BPP_ANB, CLASS_1_100, 317),
    (True, BRKGA_BPP_2R, CLASS_2_20, 10),
    (False, BRKGA_BPP_ANB, CLASS_2_20, 10),
    (True, BRKGA_BPP_2R, CLASS_2_40, 19),
    (False, BRKGA_BPP_ANB, CLASS_2_40, 19),
    (True, BRKGA_BPP_2R, CLASS_2_60, 25),
    (False, BRKGA_BPP_ANB, CLASS_2_60, 25),
    (True, BRKGA_BPP_2R, CLASS_2_80, 31),
    (False, BRKGA_BPP_ANB, CLASS_2_80, 31),
    (True, BRKGA_BPP_2R, CLASS_2_100, 39),
    (False, BRKGA_BPP_ANB, CLASS_2_100, 39),
    (True, BRKGA_BPP_2R, CLASS_3_20, 47),
    (False, BRKGA_BPP_ANB, CLASS_3_20, 51),
    (True, BRKGA_BPP_2R, CLASS_3_40, 92),
    (False, BRKGA_BPP_ANB, CLASS_3_40, 94),
    (True, BRKGA_BPP_2R, CLASS_3_60, 134),
    (False, BRKGA_BPP_ANB, CLASS_3_60, 139),
    (True, BRKGA_BPP_2R, CLASS_3_80, 182),
    (False, BRKGA_BPP_ANB, CLASS_3_80, 189),
    (True, BRKGA_BPP_2R, CLASS_3_100, 220),
    (False, BRKGA_BPP_ANB, CLASS_3_100, 223),
    (True, BRKGA_BPP_2R, CLASS_4_20, 10),
    (False, BRKGA_BPP_ANB, CLASS_4_20, 10),
    (True, BRKGA_BPP_2R, CLASS_4_40, 19),
    (False, BRKGA_BPP_ANB, CLASS_4_40, 19),
    (True, BRKGA_BPP_2R, CLASS_4_60, 23),
    (False, BRKGA_BPP_ANB, CLASS_4_60, 25),
    (True, BRKGA_BPP_2R, CLASS_4_80, 31),
    (False, BRKGA_BPP_ANB, CLASS_4_80, 31),
    (True, BRKGA_BPP_2R, CLASS_4_100, 37),
    (False, BRKGA_BPP_ANB, CLASS_4_100, 37),
    (True, BRKGA_BPP_2R, CLASS_5_20, 59),
    (False, BRKGA_BPP_ANB, CLASS_5_20, 65),
    (True, BRKGA_BPP_2R, CLASS_5_40, 114),
    (False, BRKGA_BPP_ANB, CLASS_5_40, 119),
    (True, BRKGA_BPP_2R, CLASS_5_60, 172),
    (False, BRKGA_BPP_ANB, CLASS_5_60, 180),
    (True, BRKGA_BPP_2R, CLASS_5_80, 239),
    (False, BRKGA_BPP_ANB, CLASS_5_80, 247),
    (True, BRKGA_BPP_2R, CLASS_5_100, 277),
    (False, BRKGA_BPP_ANB, CLASS_5_100, 281),
    (True, BRKGA_BPP_2R, CLASS_6_20, 10),
    (False, BRKGA_BPP_ANB, CLASS_6_20, 10),
    (True, BRKGA_BPP_2R, CLASS_6_40, 16),
    (False, BRKGA_BPP_ANB, CLASS_6_40, 16),
    (True, BRKGA_BPP_2R, CLASS_6_60, 21),
    (False, BRKGA_BPP_ANB, CLASS_6_60, 21),
    (True, BRKGA_BPP_2R, CLASS_6_80, 30),
    (False, BRKGA_BPP_ANB, CLASS_6_80, 30),
    (True, BRKGA_BPP_2R, CLASS_6_100, 32),
    (False, BRKGA_BPP_ANB, CLASS_6_100, 33),
    (True, BRKGA_BPP_2R, CLASS_7_20, 52),
    (False, BRKGA_BPP_ANB, CLASS_7_20, 55),
    (True, BRKGA_BPP_2R, CLASS_7_40, 102),
    (False, BRKGA_BPP_ANB, CLASS_7_40, 111),
    (True, BRKGA_BPP_2R, CLASS_7_60, 146),
    (False, BRKGA_BPP_ANB, CLASS_7_60, 158),
    (True, BRKGA_BPP_2R, CLASS_7_80, 208),
    (False, BRKGA_BPP_ANB, CLASS_7_80, 232),
    (True, BRKGA_BPP_2R, CLASS_7_100, 250),
    (False, BRKGA_BPP_ANB, CLASS_7_100, 271),
    (True, BRKGA_BPP_2R, CLASS_8_20, 53),
    (False, BRKGA_BPP_ANB, CLASS_8_20, 58),
    (True, BRKGA_BPP_2R, CLASS_8_40, 103),
    (False, BRKGA_BPP_ANB, CLASS_8_40, 113),
    (True, BRKGA_BPP_2R, CLASS_8_60, 147),
    (False, BRKGA_BPP_ANB, CLASS_8_60, 161),
    (True, BRKGA_BPP_2R, CLASS_8_80, 204),
    (False, BRKGA_BPP_ANB, CLASS_8_80, 224),
    (True, BRKGA_BPP_2R, CLASS_8_100, 252),
    (False, BRKGA_BPP_ANB, CLASS_8_100, 278),
    (True, BRKGA_BPP_2R, CLASS_9_20, 143),
    (False, BRKGA_BPP_ANB, CLASS_9_20, 143),
    (True, BRKGA_BPP_2R, CLASS_9_40, 275),
    (False, BRKGA_BPP_ANB, CLASS_9_40, 278),
    (True, BRKGA_BPP_2R, CLASS_9_60, 435),
    (False, BRKGA_BPP_ANB, CLASS_9_60, 437),
    (True, BRKGA_BPP_2R, CLASS_9_80, 573),
    (False, BRKGA_BPP_ANB, CLASS_9_80, 577),
    (True, BRKGA_BPP_2R, CLASS_9_100, 693),
    (False, BRKGA_BPP_ANB, CLASS_9_100, 695),
    (True, BRKGA_BPP_2R, CLASS_10_20, 41),
    (False, BRKGA_BPP_ANB, CLASS_10_20, 42),
    (True, BRKGA_BPP_2R, CLASS_10_40, 72),
    (False, BRKGA_BPP_ANB, CLASS_10_40, 74),
    (True, BRKGA_BPP_2R, CLASS_10_60, 99),
    (False, BRKGA_BPP_ANB, CLASS_10_60, 100),
    (True, BRKGA_BPP_2R, CLASS_10_80, 125),
    (False, BRKGA_BPP_ANB, CLASS_10_80, 128),
    (True, BRKGA_BPP_2R, CLASS_10_100, 154),
    (False, BRKGA_BPP_ANB, CLASS_10_100, 158),
    (True, BRKGA_BPP_2R, BENG_1_8, 54),
    (False, BRKGA_BPP_ANB, BENG_1_8, 54),
    (True, BRKGA_BPP_2R, BENG_9_10, 13),
    (False, BRKGA_BPP_ANB, BENG_9_10, 13),
    (False, PAC, CLASS_1_20, 71),
    (False, PAC, CLASS_1_40, 134),
    (False, PAC, CLASS_1_60, 200),
    (False, PAC, CLASS_1_80, 275),
    (False, PAC, CLASS_1_100, 317),
    (False, PAC, CLASS_2_20, 10),
    (False, PAC, CLASS_2_40, 19),
    (False, PAC, CLASS_2_60, 25),
    (False, PAC, CLASS_2_80, 31),
    (False, PAC, CLASS_2_100, 39),
    (False, PAC, CLASS_3_20, 51),
    (False, PAC, BENG_1_8, 54),
    (False, PAC, BENG_9_10, 13),
    (True, PAC, CLASS_1_20, 66),
    (True, PAC, CLASS_1_40, 128),
    (True, PAC, CLASS_1_60, 195),
    (True, PAC, CLASS_1_80, 270),
    (True, PAC, CLASS_1_100, 313),
    (True, PAC, CLASS_2_20, 10),
    (True, PAC, CLASS_2_40, 19),
    (True, PAC, CLASS_2_60, 25),
    (True, PAC, CLASS_2_80, 31),
    (True, PAC, CLASS_2_100, 39),
    (True, PAC, CLASS_3_20, 47),
    (True, PAC, BENG_1_8, 54),
    (True, PAC, BENG_9_10, 13),
    (True, HHANO_R, CLASS_1_20, 66),
    (True, HHANO_SR, CLASS_1_20, 66),
    (True, HHANO_R, CLASS_1_40, 131),
    (True, HHANO_SR, CLASS_1_40, 129),
    (True, HHANO_R, CLASS_1_60, 196),
    (True, HHANO_SR, CLASS_1_60, 195),
    (True, HHANO_R, CLASS_1_80, 270),
    (True, HHANO_SR, CLASS_1_80, 270),
    (True, HHANO_R, CLASS_1_100, 314),
    (True, HHANO_SR, CLASS_1_100, 313),
    (True, HHANO_R, CLASS_2_20, 10),
    (True, HHANO_SR, CLASS_2_20, 10),
    (True, HHANO_R, CLASS_2_40, 20),
    (True, HHANO_SR, CLASS_2_40, 19),
    (True, HHANO_R, CLASS_2_60, 25),
    (True, HHANO_SR, CLASS_2_60, 25),
    (True, HHANO_R, CLASS_2_80, 31),
    (True, HHANO_SR, CLASS_2_80, 31),
    (True, HHANO_R, CLASS_2_100, 39),
    (True, HHANO_SR, CLASS_2_100, 39),
    (True, HHANO_R, CLASS_3_20, 48),
    (True, HHANO_SR, CLASS_3_20, 48),
    (True, HHANO_R, CLASS_3_40, 95),
    (True, HHANO_SR, CLASS_3_40, 95),
    (True, HHANO_R, CLASS_3_60, 137),
    (True, HHANO_SR, CLASS_3_60, 137),
    (True, HHANO_R, CLASS_3_80, 186),
    (True, HHANO_SR, CLASS_3_80, 187),
    (True, HHANO_R, CLASS_3_100, 225),
    (True, HHANO_SR, CLASS_3_100, 225),
    (True, HHANO_R, CLASS_4_20, 10),
    (True, HHANO_SR, CLASS_4_20, 10),
    (True, HHANO_R, CLASS_4_40, 19),
    (True, HHANO_SR, CLASS_4_40, 19),
    (True, HHANO_R, CLASS_4_60, 25),
    (True, HHANO_SR, CLASS_4_60, 25),
    (True, HHANO_R, CLASS_4_80, 32),
    (True, HHANO_SR, CLASS_4_80, 33),
    (True, HHANO_R, CLASS_4_100, 38),
    (True, HHANO_SR, CLASS_4_100, 38),
    (True, HHANO_R, CLASS_5_20, 59),
    (True, HHANO_SR, CLASS_5_20, 59),
    (True, HHANO_R, CLASS_5_40, 116),
    (True, HHANO_SR, CLASS_5_40, 115),
    (True, HHANO_R, CLASS_5_60, 175),
    (True, HHANO_SR, CLASS_5_60, 176),
    (True, HHANO_R, CLASS_5_80, 240),
    (True, HHANO_SR, CLASS_5_80, 241),
    (True, HHANO_R, CLASS_5_100, 284),
    (True, HHANO_SR, CLASS_5_100, 284),
    (True, HHANO_R, CLASS_6_20, 10),
    (True, HHANO_SR, CLASS_6_20, 10),
    (True, HHANO_R, CLASS_6_40, 18),
    (True, HHANO_SR, CLASS_6_40, 17),
    (True, HHANO_R, CLASS_6_60, 22),
    (True, HHANO_SR, CLASS_6_60, 22),
    (True, HHANO_R, CLASS_6_80, 30),
    (True, HHANO_SR, CLASS_6_80, 30),
    (True, HHANO_R, CLASS_6_100, 34),
    (True, HHANO_SR, CLASS_6_100, 34),
    (True, HHANO_R, CLASS_7_20, 52),
    (True, HHANO_SR, CLASS_7_20, 52),
    (True, HHANO_R, CLASS_7_40, 106),
    (True, HHANO_SR, CLASS_7_40, 107),
    (True, HHANO_R, CLASS_7_60, 152),
    (True, HHANO_SR, CLASS_7_60, 153),
    (True, HHANO_R, CLASS_7_80, 216),
    (True, HHANO_SR, CLASS_7_80, 217),
    (True, HHANO_R, CLASS_7_100, 260),
    (True, HHANO_SR, CLASS_7_100, 259),
    (True, HHANO_R, CLASS_8_20, 53),
    (True, HHANO_SR, CLASS_8_20, 53),
    (True, HHANO_R, CLASS_8_40, 106),
    (True, HHANO_SR, CLASS_8_40, 105),
    (True, HHANO_R, CLASS_8_60, 155),
    (True, HHANO_SR, CLASS_8_60, 154),
    (True, HHANO_R, CLASS_8_80, 213),
    (True, HHANO_SR, CLASS_8_80, 214),
    (True, HHANO_R, CLASS_8_100, 261),
    (True, HHANO_SR, CLASS_8_100, 262),
    (True, HHANO_R, CLASS_9_20, 143),
    (True, HHANO_SR, CLASS_9_20, 143),
    (True, HHANO_R, CLASS_9_40, 275),
    (True, HHANO_SR, CLASS_9_40, 275),
    (True, HHANO_R, CLASS_9_60, 435),
    (True, HHANO_SR, CLASS_9_60, 435),
    (True, HHANO_R, CLASS_9_80, 573),
    (True, HHANO_SR, CLASS_9_80, 573),
    (True, HHANO_R, CLASS_9_100, 693),
    (True, HHANO_SR, CLASS_9_100, 693),
    (True, HHANO_R, CLASS_10_20, 41),
    (True, HHANO_SR, CLASS_10_20, 41),
    (True, HHANO_R, CLASS_10_40, 73),
    (True, HHANO_SR, CLASS_10_40, 73),
    (True, HHANO_R, CLASS_10_60, 101),
    (True, HHANO_SR, CLASS_10_60, 101),
    (True, HHANO_R, CLASS_10_80, 129),
    (True, HHANO_SR, CLASS_10_80, 130),
    (True, HHANO_R, CLASS_10_100, 161),
    (True, HHANO_SR, CLASS_10_100, 162),
    __lb_avg_denormalize(True, MXGA, CLASS_1_20, 1.03),
    __lb_avg_denormalize(True, MXGA, CLASS_1_40, 1.04),
    __lb_avg_denormalize(True, MXGA, CLASS_1_60, 1.04, 195),  # 19.4
    __lb_avg_denormalize(True, MXGA, CLASS_1_80, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_1_100, 1.02, 313),  # 31.2
    __lb_avg_denormalize(True, MXGA, CLASS_2_20, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_2_40, 1.1),
    __lb_avg_denormalize(True, MXGA, CLASS_2_60, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_2_80, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_2_100, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_3_20, 1.04),
    __lb_avg_denormalize(True, MXGA, CLASS_3_40, 1.09),
    __lb_avg_denormalize(True, MXGA, CLASS_3_60, 1.08),
    __lb_avg_denormalize(True, MXGA, CLASS_3_80, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_3_100, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_4_20, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_4_40, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_4_60, 1.10),
    __lb_avg_denormalize(True, MXGA, CLASS_4_80, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_4_100, 1.03),
    __lb_avg_denormalize(True, MXGA, CLASS_5_20, 1.04),
    __lb_avg_denormalize(True, MXGA, CLASS_5_40, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_5_60, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_5_80, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_5_100, 1.05),
    __lb_avg_denormalize(True, MXGA, CLASS_6_20, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_6_40, 1.4),
    __lb_avg_denormalize(True, MXGA, CLASS_6_60, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_6_80, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_6_100, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_7_20, 1.11),
    __lb_avg_denormalize(True, MXGA, CLASS_7_40, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_7_60, 1.05),
    __lb_avg_denormalize(True, MXGA, CLASS_7_80, 1.08),
    __lb_avg_denormalize(True, MXGA, CLASS_7_100, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_8_20, 1.10),
    __lb_avg_denormalize(True, MXGA, CLASS_8_40, 1.09),
    __lb_avg_denormalize(True, MXGA, CLASS_8_60, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_8_80, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_8_100, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_9_20, 1.0),
    __lb_avg_denormalize(True, MXGA, CLASS_9_40, 1.01),
    __lb_avg_denormalize(True, MXGA, CLASS_9_60, 1.01),
    __lb_avg_denormalize(True, MXGA, CLASS_9_80, 1.01),
    __lb_avg_denormalize(True, MXGA, CLASS_9_100, 1.01),
    __lb_avg_denormalize(True, MXGA, CLASS_10_20, 1.13),
    __lb_avg_denormalize(True, MXGA, CLASS_10_40, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_10_60, 1.07),
    __lb_avg_denormalize(True, MXGA, CLASS_10_80, 1.06),
    __lb_avg_denormalize(True, MXGA, CLASS_10_100, 1.04),
    (False, EALGFI, CLASS_1_20, 71),
    (False, EALGFI, CLASS_1_40, 134),
    (False, EALGFI, CLASS_1_60, 200),
    (False, EALGFI, CLASS_1_80, 275),
    (False, EALGFI, CLASS_1_100, 317),
    (False, EALGFI, CLASS_2_20, 10),
    (False, EALGFI, CLASS_2_40, 19),
    (False, EALGFI, CLASS_2_60, 25),
    (False, EALGFI, CLASS_2_80, 31),
    (False, EALGFI, CLASS_2_100, 39),
    (False, EALGFI, CLASS_3_20, 51),
    (False, EALGFI, CLASS_3_40, 94),
    (False, EALGFI, CLASS_3_60, 139),
    (False, EALGFI, CLASS_3_80, 189),
    (False, EALGFI, CLASS_3_100, 224),
    (False, EALGFI, CLASS_4_20, 10),
    (False, EALGFI, CLASS_4_40, 19),
    (False, EALGFI, CLASS_4_60, 23),
    (False, EALGFI, CLASS_4_80, 31),
    (False, EALGFI, CLASS_4_100, 37),
    (False, EALGFI, CLASS_5_20, 65),
    (False, EALGFI, CLASS_5_40, 119),
    (False, EALGFI, CLASS_5_60, 180),
    (False, EALGFI, CLASS_5_80, 247),
    (False, EALGFI, CLASS_5_100, 284),
    (False, EALGFI, CLASS_6_20, 10),
    (False, EALGFI, CLASS_6_40, 17),
    (False, EALGFI, CLASS_6_60, 21),
    (False, EALGFI, CLASS_6_80, 30),
    (False, EALGFI, CLASS_6_100, 32),
    (False, EALGFI, CLASS_7_20, 55),
    (False, EALGFI, CLASS_7_40, 111),
    (False, EALGFI, CLASS_7_60, 159),
    (False, EALGFI, CLASS_7_80, 232),
    (False, EALGFI, CLASS_7_100, 271),
    (False, EALGFI, CLASS_8_20, 58),
    (False, EALGFI, CLASS_8_40, 113),
    (False, EALGFI, CLASS_8_60, 161),
    (False, EALGFI, CLASS_8_80, 224),
    (False, EALGFI, CLASS_8_100, 277),
    (False, EALGFI, CLASS_9_20, 143),
    (False, EALGFI, CLASS_9_40, 278),
    (False, EALGFI, CLASS_9_60, 437),
    (False, EALGFI, CLASS_9_80, 577),
    (False, EALGFI, CLASS_9_100, 695),
    (False, EALGFI, CLASS_10_20, 42),
    (False, EALGFI, CLASS_10_40, 74),
    (False, EALGFI, CLASS_10_60, 101),
    (False, EALGFI, CLASS_10_80, 128),
    (False, EALGFI, CLASS_10_100, 160),
    (False, GRASP_VND, CLASS_1_20, 71),
    (False, GRASP_VND, CLASS_1_40, 134),
    (False, GRASP_VND, CLASS_1_60, 200),
    (False, GRASP_VND, CLASS_1_80, 275),
    (False, GRASP_VND, CLASS_1_100, 317),
    (False, GRASP_VND, CLASS_2_20, 10),
    (False, GRASP_VND, CLASS_2_40, 19),
    (False, GRASP_VND, CLASS_2_60, 25),
    (False, GRASP_VND, CLASS_2_80, 31),
    (False, GRASP_VND, CLASS_2_100, 39),
    (False, GRASP_VND, CLASS_3_20, 51),
    (False, GRASP_VND, CLASS_3_40, 94),
    (False, GRASP_VND, CLASS_3_60, 139),
    (False, GRASP_VND, CLASS_3_80, 189),
    (False, GRASP_VND, CLASS_3_100, 223),
    (False, GRASP_VND, CLASS_4_20, 10),
    (False, GRASP_VND, CLASS_4_40, 19),
    (False, GRASP_VND, CLASS_4_60, 25),
    (False, GRASP_VND, CLASS_4_80, 31),
    (False, GRASP_VND, CLASS_4_100, 38),
    (False, GRASP_VND, CLASS_5_20, 65),
    (False, GRASP_VND, CLASS_5_40, 119),
    (False, GRASP_VND, CLASS_5_60, 180),
    (False, GRASP_VND, CLASS_5_80, 247),
    (False, GRASP_VND, CLASS_5_100, 282),
    (False, GRASP_VND, CLASS_6_20, 10),
    (False, GRASP_VND, CLASS_6_40, 17),
    (False, GRASP_VND, CLASS_6_60, 21),
    (False, GRASP_VND, CLASS_6_80, 30),
    (False, GRASP_VND, CLASS_6_100, 34),
    (False, GRASP_VND, CLASS_7_20, 55),
    (False, GRASP_VND, CLASS_7_40, 111),
    (False, GRASP_VND, CLASS_7_60, 159),
    (False, GRASP_VND, CLASS_7_80, 232),
    (False, GRASP_VND, CLASS_7_100, 271),
    (False, GRASP_VND, CLASS_8_20, 58),
    (False, GRASP_VND, CLASS_8_40, 113),
    (False, GRASP_VND, CLASS_8_60, 161),
    (False, GRASP_VND, CLASS_8_80, 224),
    (False, GRASP_VND, CLASS_8_100, 278),
    (False, GRASP_VND, CLASS_9_20, 143),
    (False, GRASP_VND, CLASS_9_40, 278),
    (False, GRASP_VND, CLASS_9_60, 437),
    (False, GRASP_VND, CLASS_9_80, 577),
    (False, GRASP_VND, CLASS_9_100, 695),
    (False, GRASP_VND, CLASS_10_20, 42),
    (False, GRASP_VND, CLASS_10_40, 74),
    (False, GRASP_VND, CLASS_10_60, 100),
    (False, GRASP_VND, CLASS_10_80, 129),
    (False, GRASP_VND, CLASS_10_100, 159),
    (False, GRASP_VND, BENG_1_8, 54),
    (False, GRASP_VND, BENG_9_10, 13),
)


def get_related_work(
        with_rotation: bool | None = None,
        without_rotation: bool | None = None,
        algo_select: Callable[[Element], bool] =
        lambda _: True,
        inst_group_select: Callable[[Element], bool] =
        lambda _: True) -> tuple[
        tuple[bool, Element, Element, int], ...]:
    """
    Get the related work of a given type.

    :param with_rotation: include the data with rotation
    :param without_rotation: include the data without rotation
    :param algo_select: the algorithm selector
    :param inst_group_select: the instance group selector
    :return: An iterable with the related works
    """
    res: Iterable[tuple[bool, Element, Element, int]]
    if (with_rotation is None) and (without_rotation is None):
        res = __RW__AVERAGE
    else:
        if with_rotation is None:
            with_rotation = not without_rotation
        if without_rotation is None:
            without_rotation = not with_rotation

        if without_rotation and with_rotation:
            res = __RW__AVERAGE
        elif with_rotation:
            res = (x for x in __RW__AVERAGE if x[0])
        elif without_rotation:
            res = (x for x in __RW__AVERAGE if not x[0])
        else:
            return ()
    return tuple(sorted(filter(
        lambda x: algo_select(x[1]) and inst_group_select(x[2]), res),
        key=lambda v: (v[0], INST_GROUP_SORT_KEY(v[2]), v[1], v[3])))


def make_comparison_table_data(
        data: Iterable[PackingResult],
        with_rotation: bool,
        algo_from_pr: Callable[[PackingResult], Element] = lambda x: Element(
            x.end_result.algorithm, x.end_result.objective),
        algo_sort_key: Callable[[Element], Any] = lambda x: (
        -1 if (x.reference == NO_REF) else 1, x),
        rw_algo_selector: Callable[[Element], bool] = lambda _: True,
        aggregator: Callable[[Iterable[int | float]], int | float] = mean,
        additional: Callable[[Element], Iterable[tuple[
            Element, Callable[[Iterable[int | float]], int | float]]]] =
        lambda _: ()) -> tuple[tuple[Element, ...], tuple[tuple[
        Element, tuple[int | float | None, ...]], ...]]:
    """
    Create the data for an end result comparison table.

    :param data: the source data
    :param with_rotation: are we doing stuff with rotation?
    :param algo_from_pr: convert a packing result to an algorithm
    :param algo_sort_key: the algorithm sort key
    :param rw_algo_selector: the related work algorithm selector
    :param aggregator: the routine for per-instance averaging of bins
    :param additional: an additional column constructor

    :return: the table data: the title row columns followed by the data
        row-by-row, each row leading with an instance group identifier
    """
    per_algo_data: dict[Element, tuple[Callable[[
        Iterable[int | float]], int | float], dict[str, list[int]]]] = {}
    for pr in data:
        algo: Element = algo_from_pr(pr)
        inst_dict: dict[str, list[int]]
        if algo in per_algo_data:
            inst_dict = per_algo_data[algo][1]
        else:
            inst_dict = {}
            per_algo_data[algo] = aggregator, inst_dict
        inst: str = pr.end_result.instance
        val: int = check_int_range(
            pr.objectives[BIN_COUNT_NAME], BIN_COUNT_NAME, 1, 1_000_000_000)
        if inst in inst_dict:
            inst_dict[inst].append(val)
        else:
            inst_dict[inst] = [val]

    groups: Final[list[Element]] = sorted({
        INSTANCES_TO_GROUPS[k] for kk in per_algo_data.values()
        for k in kk[1]}, key=INST_GROUP_SORT_KEY)

    # get the averages
    algo_per_group_data: Final[dict[Element, dict[Element, int | float]]] = {}
    for algo, inst_dict_and_avg in per_algo_data.items():
        dothis: list[tuple[Element, Callable[[
            Iterable[int | float]], int | float]]] = [
            (algo, inst_dict_and_avg[0])]
        dothis.extend(additional(algo))

        for xyz in dothis:
            avg: Callable[[Iterable[int | float]], int | float] = xyz[1]
            inst_dict = inst_dict_and_avg[1]
            ag: dict[Element, int | float] = {}
            algo_per_group_data[xyz[0]] = ag
            algo_runs: int | None = None
            for g in groups:
                gis: tuple[str, ...] = GROUPS_TO_INSTANCES[g]
                gisd: list[list[int | float]] = []
                ok: bool = True
                for inst in gis:
                    if inst not in inst_dict:
                        ok = False
                    gisd.append(cast("list[int | float]", inst_dict[inst]))
                if ok:
                    if list.__len__(gisd) <= 0:
                        raise ValueError(f"Empty group {g!r} for {algo!r}?")
                    if list.__len__(gisd) != tuple.__len__(gis):
                        raise ValueError(
                            f"Some data missing in {g!r} for {algo!r}?")
                    ll = list.__len__(gisd[0])
                    if algo_runs is None:
                        algo_runs = ll
                    elif algo_runs != ll:
                        raise ValueError(
                            f"Inconsistent number of runs for {algo!r}: "
                            f"{ll} in {gis[0]!r} vs. {algo_runs}.")
                    if not all(list.__len__(xxx) == ll for xxx in gisd):
                        raise ValueError(
                            f"Inconsistent run numbers in {g!r} for {algo!r}.")
                    ag[g] = round(sum(cast("Iterable", (
                        try_int(avg(v)) for v in gisd))))
                elif len(gisd) > 0:
                    raise ValueError(f"Incomplete group {g!r} for {algo!r}.")

    rw: tuple[tuple[bool, Element, Element, float], ...] = get_related_work(
        with_rotation=with_rotation, without_rotation=not with_rotation,
        algo_select=rw_algo_selector, inst_group_select=groups.__contains__)

    algorithms: list[Element] = list({rww[1] for rww in rw})
    algorithms.sort(key=algo_sort_key)
    our_algorithms = sorted(algo_per_group_data.keys(), key=algo_sort_key)

    rows: list[tuple[Element, tuple[int | float | None, ...]]] = []
    for g in groups:
        xdata: list[int | float | None] = []
        for algo in algorithms:
            found: bool = False
            for zz in rw:
                if zz[1] != algo:
                    continue
                if zz[2] != g:
                    continue
                found = True
                xdata.append(zz[3])
                break
            if found:
                continue
            xdata.append(None)
        for algo in our_algorithms:
            agz: dict[Element, int | float] = algo_per_group_data[algo]
            if g not in agz:
                xdata.append(None)
                continue
            xdata.append(agz[g])
        rows.append((g, tuple(xdata)))

    algorithms.extend(our_algorithms)
    return tuple(algorithms), tuple(rows)


def make_comparison_table(
        dest: Callable[[str], Any],
        data: tuple[tuple[Element, ...], tuple[tuple[Element, tuple[
            int | float | None, ...]], ...]],
        name_to_strs: Callable[[Element], tuple[str, str]] = lambda s: (
        s.name.split("_")[0], s.name_suffix),
        format_best: Callable[[str], str] =
        lambda s: f"\\textbf{{{s}}}",
        count_best_over: Iterable[tuple[Iterable[Element], str]] = ()) -> None:
    """
    Make a comparison table in LaTeX.

    :param dest: the destination
    :param data: the source data
    :param name_to_strs: the converter of names to strings
    :param format_best: the formatter for best values
    :param count_best_over: a set of instances to include in the best counting
        and a corresponding label
    """
    inst_names: Final[list[tuple[str, str]]] = [
        name_to_strs(d[0]) for d in data[1]]
    inst_cols: int = 2 if any(
        str.__len__(n[1]) > 0 for n in inst_names) else 1

    dest("% begin auto-generated LaTeX table comparing "
         "end results with related work.%")
    writer: list[str] = [r"\begin{tabular}{@{}r"]
    if inst_cols > 1:
        writer.append("@{}l")
    writer.extend("r" for _ in range(tuple.__len__(data[0])))
    writer.append("@{}}%")
    dest("".join(writer))
    writer.clear()
    dest(r"\hline%")

    head_names: Final[list[tuple[str, str]]] = [
        name_to_strs(d) for d in data[0]]
    head_names.insert(0, ("instance", "group"))
    if inst_cols > 1:
        head_names.insert(0, ("instance", "group"))
    head_count: Final[int] = list.__len__(head_names)

    for dim in (0, 1):
        i = 0
        while i < head_count:
            head = head_names[i]
            next_i = i + 1
            while (next_i < head_count) and (
                    head_names[next_i][dim] == head[dim]):
                next_i += 1

            col_txt: str = f"\\multicolumn{{{next_i - i}}}{{"
            if i <= 0:
                col_txt = f"{col_txt}@{{}}"
            col_txt = f"{col_txt}c"
            if next_i >= head_count:
                col_txt = f"{col_txt}@{{}}"
            col_txt = f"{col_txt}}}{{{head[dim]}}}"
            if next_i >= head_count:
                col_txt = f"{col_txt}\\\\%"
            writer.append(col_txt)
            i = next_i
        dest("&".join(writer))
        writer.clear()
    last_first_name = ""

    count_best: list[tuple[str, list[int]]] = []
    count_as: dict[Element, list[list[int]]] = {}
    for count_this, title in count_best_over:
        lst: list[int] = [0] * tuple.__len__(data[0])
        count_best.append((title, lst))
        for k in count_this:
            if k in count_as:
                count_as[k].append(lst)
            else:
                count_as[k] = [lst]

    for i, data_row in enumerate(data[1]):
        inst_name: tuple[str, str] = inst_names[i]
        if inst_name[0] != last_first_name:
            last_first_name = inst_name[0]
            dest(r"\hline%")
        writer.append(last_first_name)
        if inst_cols > 1:
            in1: str = str.strip(inst_name[1])
            if str.__len__(in1) > 0:
                in1 = str.strip(f"/{in1}")
                if str.__len__(in1) <= 1:
                    in1 = ""
            writer.append(in1)

        row_data: tuple[int | float | None, ...] = data_row[1]
        minimum: int | float = inf
        for ddd in row_data:
            if (ddd is not None) and (ddd < minimum):
                minimum = ddd
        if not isfinite(minimum):
            raise ValueError(f"Huuhhh? {inst_name} has non-finite min?")
        instance: Element = data_row[0]
        for iij, ddd in enumerate(row_data):
            if ddd is None:
                writer.append("")
                continue
            writer.append(
                format_best(str(ddd)) if ddd <= minimum else str(ddd))
            if (ddd <= minimum) and (instance in count_as):
                for lst in count_as[instance]:
                    lst[iij] += 1
        dest(f"{'&'.join(writer)}\\\\%")
        writer.clear()

    dest(r"\hline%")
    if list.__len__(count_best) > 0:
        for brow in count_best:
            writer.append(
                f"\\multicolumn{{{inst_cols}}}{{@{{}}r}}{{{brow[0]}}}")
            writer.extend(map(str, brow[1]))
            writer[-1] = f"{writer[-1]}\\\\%"
            dest("&".join(writer))
        dest(r"\hline%")
    dest(r"\end{tabular}%")
    dest("% end auto-generated LaTeX table comparing "
         "end results with related work.%")
