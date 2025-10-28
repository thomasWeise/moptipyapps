"""
Obtain Instances of the 2D-dimensional Bin Packing Problem.

With this program, we can obtain the instances of the two-dimensional bin
packing problem and convert them to a singular resource file.
The resource file holds one instance per line.

1. Manuel Iori, Vinícius Loti de Lima, Silvano Martello, and Michele Monaci.
   *2DPackLib*.
   https://site.unibo.it/operations-research/en/research/2dpacklib
2. Manuel Iori, Vinícius Loti de Lima, Silvano Martello, and Michele
   Monaci. 2DPackLib: A Two-Dimensional Cutting and Packing Library.
   *Optimization Letters* 16(2):471-480. March 2022.
   https://doi.org/10.1007/s11590-021-01808-y
"""

import zipfile
from io import BytesIO
from math import isqrt
from os.path import dirname, exists
from typing import Any, Callable, Final, Iterable

# noinspection PyPackageRequirements
import certifi  # type: ignore

# noinspection PyPackageRequirements
import urllib3  # type: ignore
from pycommons.io.path import Path, directory_path, file_path, write_lines
from pycommons.io.temp import temp_dir
from pycommons.types import type_error

import moptipyapps.binpacking2d.instance as inst_mod
from moptipyapps.binpacking2d.instance import (
    INSTANCES_RESOURCE,
    Instance,
)

#: the base url for 2DPackLib
__BASE_URL: str = \
    "https://site.unibo.it/operations-research/en/research/2dpacklib"

#: The base URLs of the relevant 2DPackLib instances.
__BASE_URLS: Final[Iterable[str]] = tuple(
    f"{__BASE_URL}/{f}.zip" for f in ["a", "beng", "class"])


def download_2dpacklib_instances(
        dest_dir: str,
        source_urls: Iterable[str] = __BASE_URLS,
        http: urllib3.PoolManager = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())) \
        -> Iterable[Path]:
    """
    Download the instances from 2DPackLib to a folder.

    This function downloads the instances from 2DPackLib, which are provided
    as zip archives containing one file per instance. It will extract all the
    instances into the folder `dest_dir` and return a tuple of the extracted
    files. You can specify the source URLs of the zip archives if you want,
    but by default we use the three instance sets `A`, `BENG`, and `CLASS`.

    :param dest_dir: the destination directory
    :param source_urls: the source URLs from which to download the zip
        archives with the 2DPackLib-formatted instances
    :param http: the HTTP pool
    :return: the list of unpackaged files
    """
    dest: Final[Path] = directory_path(dest_dir)
    if not isinstance(source_urls, Iterable):
        raise type_error(source_urls, "source_urls", Iterable)
    result: Final[list[Path]] = []

    for i, url in enumerate(source_urls):  # iterate over the source URLs
        if not isinstance(url, str):
            raise type_error(url, f"source_urls[{i}]", str)

        response = http.request(  # download the zip archive
            "GET", url, redirect=True, retries=30)
        with zipfile.ZipFile(BytesIO(response.data), mode="r") as z:
            # unzip the ins2D files from the archives
            files: Iterable[str] = [f for f in z.namelist()
                                    if f.endswith(".ins2D")]
            paths: list[Path] = [dest.resolve_inside(f) for f in files]
            for path in paths:
                if exists(path):
                    raise ValueError(f"file {path} already exists!")
            z.extractall(dest, members=files)
            for path in paths:
                path.enforce_file()
                result.append(path)
    result.sort()
    return tuple(result)


def __normalize_2dpacklib_inst_name(instname: str) -> str:
    """
    Normalize an instance name.

    :param instname: the name
    :return: the normalized name
    """
    if not isinstance(instname, str):
        raise type_error(instname, "name", str)
    instname = instname.strip().lower()
    if (len(instname) == 2) and \
            (instname[0] in "abcdefghijklmnoprstuvwxyz") and \
            (instname[1] in "123456789"):
        return f"{instname[0]}0{instname[1]}"
    if instname.startswith("cl_"):
        return f"cl{instname[3:]}"
    return instname


def append_almost_squares_strings(collector: Callable[[
        tuple[str, str]], Any]) -> None:
    """
    Append the strings of the almost squares instances.

    :param collector: the instance collector
    :return: the strings
    """
    objects: list[list[int]] = [[2, 1, 1]]
    size: int = 2
    for small_side in range(2, 36):
        big_side = small_side + 1
        objects.append([big_side, small_side, 1])
        size += small_side * big_side

        bin_small = isqrt(size)
        bin_big = bin_small + 1
        if (bin_big * bin_small) == size:
            iname: str = str(small_side)
            iname = f"asqas{iname}" if len(iname) >= 2 else f"asqas0{iname}"
            collector((iname, Instance(
                iname, bin_big, bin_small, objects).to_compact_str()))


def join_instances_to_compact(
        binpacklib2d_files: Iterable[str], dest_file: str,
        normalizer: Callable[[str], str] = __normalize_2dpacklib_inst_name) \
        -> tuple[Path, Iterable[str]]:
    """
    Join all instances from a set of 2DPackLib files to one compact file.

    :param binpacklib2d_files: the iterable of 2DPackLib file paths
    :param dest_file: the destination file
    :param normalizer: the name normalizer, i.e., a function that processes
        and/or transforms an instance name
    :return: the canonical destination path and the list of instance names
        stored
    """
    if not isinstance(binpacklib2d_files, Iterable):
        raise type_error(binpacklib2d_files, "files", Iterable)
    if not callable(normalizer):
        raise type_error(normalizer, "normalizer", call=True)
    dest_path = Path(dest_file)
    data: Final[list[tuple[str, str]]] = []
    for file in binpacklib2d_files:
        inst: Instance = Instance.from_2dpacklib(file_path(file))
        inst.name = normalizer(inst.name)
        data.append((inst.name, inst.to_compact_str()))
    append_almost_squares_strings(data.append)  # add the asquas instances
    data.sort()
    with dest_path.open_for_write() as wd:
        write_lines((content for _, content in data), wd)
    dest_path.enforce_file()
    return dest_path, [thename for thename, _ in data]


def make_2dpacklib_resource(
        dest_file: "str | None" = None,
        source_urls: Iterable[str] = __BASE_URLS,
        normalizer: Callable[[str], str] = __normalize_2dpacklib_inst_name)\
        -> tuple[Path, Iterable[str]]:
    """
    Make the resource with all the relevant 2DPackLib instances.

    :param dest_file: the optional path to the destination file
    :param source_urls: the source URLs from which to download the zip
        archives with the 2DPackLib-formatted instances
    :param normalizer: the name normalizer, i.e., a function that processes
        and/or transforms an instance name
    :return: the canonical path to the and the list of instance names stored
    """
    dest_path: Final[Path] = directory_path(dirname(inst_mod.__file__))\
        .resolve_inside(INSTANCES_RESOURCE) if dest_file is None \
        else Path(dest_file)
    with temp_dir() as temp:
        files: Iterable[Path] = download_2dpacklib_instances(
            dest_dir=temp, source_urls=source_urls)
        return join_instances_to_compact(
            files, dest_path, normalizer)


# create the tables if this is the main script
if __name__ == "__main__":
    _, names = make_2dpacklib_resource()
    rows: list[str] = ["_INSTANCES: Final[tuple[str, ...]] = ("]
    current = "    "
    has_space: bool = True
    for name in names:
        if (len(current) + (3 if has_space else 4) + len(name)) > 78:
            rows.append(current)
            current = "    "
            has_space = True
        current = f'{current}"{name}",' if has_space else \
            f'{current} "{name}",'
        has_space = False
    rows.append(current[:-1] + ")")

    cmt_strs: list[str] = (
        "the the list of instance names of the 2DPackLib bin "
        f"packing set downloaded from {__BASE_URL} ('a*',"
        "'beng*', 'cl*') as well as the four non-trivial "
        "'Almost Squares in Almost Squares' instances ('asqas*').").split()
    cmt_str = "#:"
    for word in cmt_strs:
        if len(word) + 1 + len(cmt_str) <= 79:
            cmt_str = f"{cmt_str} {word}"
        else:
            print(cmt_str)  # noqa
            cmt_str = f"#: {word}"
    if len(cmt_str) > 2:
        print(cmt_str)  # noqa

    for s in rows:
        print(s)  # noqa
