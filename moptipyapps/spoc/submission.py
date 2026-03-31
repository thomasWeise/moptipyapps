"""
Create submission data for the SPOC challenge.

>>> from moptipy.spaces.permutations import Permutations
>>> perms = Permutations((0, 1, 2, 3))
>>> submit = SubmissionSpace(perms, "The Challenge", "The Problem",
...                          "The Title", "The Description")
>>> submit.initialize()
>>> the_x = submit.create()
>>> the_x[:] = perms.blueprint[:]
>>> the_str = submit.to_str(the_x)
>>> print(the_str)
0;1;2;3
<BLANKLINE>
----------- SUBMISSION -----------
<BLANKLINE>
[
      {
            "decisionVector": [
                  0,
                  1,
                  2,
                  3
            ],
            "problem": "The Problem",
            "challenge": "The Challenge",
            "name": "The Title",
            "description": "The Description"
      }
]

>>> the_x2 = submit.from_str(the_str)
>>> submit.is_equal(the_x, the_x2)
True
"""

from io import StringIO
from json import dump
from typing import Any, Final

import numpy as np
from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueLogSection
from pycommons.math.int_math import try_int
from pycommons.strings.chars import NEWLINE
from pycommons.types import type_error


def __listify(x: Any) -> list | int | float:
    """
    Convert a solution to a list.

    :param x: the x value
    :return: the list
    """
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return try_int(x)
    if isinstance(x, np.number):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return try_int(float(x))
        raise type_error(x, "x", (np.integer, np.floating))
    if isinstance(x, np.ndarray):
        x = np.array(x).tolist()
    elif not isinstance(x, list):
        x = list(x)
    for i, y in enumerate(x):
        x[i] = __listify(y)
    return x


def _check_str(s: str | None) -> str | None:
    """
    Check a string.

    :param s: the string
    :return: the fixed string
    """
    use: str | None = s
    if use is not None:
        use = str.strip(use)
        if str.__len__(use) <= 0:
            return None
        for c in NEWLINE:
            if c in use:
                raise ValueError(f"{s=} forbidden, contains {c!r}.")
    return use


def to_submission(challenge_id: str,
                  problem_id: str,
                  x: Any,
                  title: str | None = None,
                  description: str | None = None) -> str:
    """
    Create a submission file text.

    This function follows the specification given in
    <https://api.optimize.esa.int/data/tools/submission_helper.py>, with the
    exception that it tries to convert floating points to integers, where
    possible without loss of precision.

    :param challenge_id: a string of the challenge identifier (found on the
        corresponding problem page)
    :param problem_id: a string of the problem identifier (found on the
        corresponding problem page)
    :param x: the result data
    :param title: a string that can be used to give your submission a title
    :param description: a string that can contain meta-information about your
        submission

    >>> print(to_submission("a", "b", (1, 2, 3), "c", "d"))
    [
          {
                "decisionVector": [
                      1,
                      2,
                      3
                ],
                "problem": "b",
                "challenge": "a",
                "name": "c",
                "description": "d"
          }
    ]

    >>> print(to_submission("a", "b", np.array(((1, 2, 3), (0.2, 4, 4.3))),
    ...         "The Title", "The Description"))
    [
          {
                "decisionVector": [
                      [
                            1,
                            2,
                            3
                      ],
                      [
                            0.2,
                            4,
                            4.3
                      ]
                ],
                "problem": "b",
                "challenge": "a",
                "name": "The Title",
                "description": "The Description"
          }
    ]
    """
    cid: str | None = _check_str(challenge_id)
    if cid is None:
        raise ValueError(f"{challenge_id=}")
    pid: str | None = _check_str(problem_id)
    if pid is None:
        raise ValueError(f"{problem_id=}")
    title = _check_str(title)
    description = _check_str(description)

    # converting numpy datatypes to python datatypes
    x = __listify(x)

    with StringIO() as io:
        dump([{"decisionVector": x,
               "problem": pid,
               "challenge": cid,
               "name": "" if title is None else title,
               "description": "" if description is None else description}],
             fp=io, indent=6)
        return str.strip(io.getvalue())


#: the inner submission separator
_SUBMISSION_SEPARATOR_INNER: Final[str] = "----------- SUBMISSION -----------"
#: the submission separator
_SUBMISSION_SEPARATOR: Final[str] = f"\n\n{_SUBMISSION_SEPARATOR_INNER}\n\n"


class SubmissionSpace(Space):
    """
    A space that also provides the submission data.

    This space is designed to wrap around an existing space type and to
    generate the SPOC submission text when textifying space elements.
    """

    def __init__(self, space: Space,
                 challenge_id: str,
                 problem_id: str,
                 title: str | None = None,
                 description: str | None = None) -> None:
        """
        Create a submission wrapper space.

        :param space: the space to wrap
        :param challenge_id: the challenge identifier
        :param problem_id: the problem identifier
        :param title: the title string, if any
        :param description: the description string, if any
        """
        self.create = space.create  # type: ignore
        self.copy = space.copy  # type: ignore
        self.is_equal = space.is_equal  # type: ignore
        self.validate = space.validate  # type: ignore
        self.n_points = space.n_points  # type: ignore
        self.initialize = space.initialize  # type: ignore

        #: the internal space copy
        self.space: Final[Space] = space

        pid = _check_str(problem_id)
        if pid is None:
            raise ValueError(f"{problem_id=}")
        #: the problem ID
        self.problem_id: Final[str] = pid
        cid = _check_str(challenge_id)
        if cid is None:
            raise ValueError(f"{challenge_id=}")
        #: the challenge ID
        self.challenge_id: Final[str] = cid
        #: the title
        self.title: Final[str | None] = _check_str(title)
        #: the description
        self.description: Final[str | None] = _check_str(description)

    def to_str(self, x) -> str:  # +book
        """
        Obtain a textual representation of an instance of the data structure.

        :param x: the instance
        :return: the string representation of x
        """
        raw: Final[str] = str.strip(self.space.to_str(x))
        if _SUBMISSION_SEPARATOR_INNER in raw:
            raise ValueError(
                f"{_SUBMISSION_SEPARATOR_INNER} not allowed in text.")
        sub: Final[str] = to_submission(self.challenge_id, self.problem_id, x,
                                        self.title, self.description)
        return f"{raw}{_SUBMISSION_SEPARATOR}{sub}"

    def from_str(self, text: str) -> Any:  # +book
        """
        Transform a string `text` to one element of the space.

        :param text: the input string
        :return: the element in the space corresponding to `text`
        """
        i = str.find(text, _SUBMISSION_SEPARATOR_INNER)
        if i > 0:
            text = str.strip(text[:i])
        return self.space.from_str(text)

    def __str__(self) -> str:
        """
        Get the submission space ID.

        :return: the submission space ID
        """
        return f"s_{self.space.__str__()}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters to a logger.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value("problemId", self.problem_id)
        logger.key_value("challengeId", self.challenge_id)
        logger.key_value("title", self.title)
        logger.key_value("description", self.description)
        with logger.scope("i") as i:
            self.space.log_parameters_to(i)
