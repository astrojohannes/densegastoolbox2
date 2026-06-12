#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAM-aware model-grid loader for Dense Gas Toolbox 2.

Key points:
- Safe concatenated-pickle-stream reader for old pandas pickle streams.
- Reduces every chunk immediately to the user-requested columns/rows before
  writing temporary chunk files.
- Checks available RAM before expensive concat operations and aborts with a
  helpful error instead of letting the OS kill the process.

Environment variables for tuning:
- DGT_CHUNK_TARGET_RAM_GB: target in-memory chunk size before writing a reduced
  chunk. Default: min(1.0 GB, 10% of available RAM, but at least 0.1 GB).
- DGT_MIN_FREE_RAM_GB: RAM that should remain free. Default: 2.0 GB.
- DGT_CONCAT_RAM_FACTOR: multiplier for temporary RAM required during concat.
  Default: 3.0.
- DGT_MAX_PICKLE_OBJECTS: optional safety limit for serialized objects read.
- DGT_TMPDIR: temporary directory. Default: tmp.
"""

import gc
import glob
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psutil
import requests

try:
    # Pandas compatibility unpickler for old pandas pickle objects.
    # Important: we use it at the current file position, never via pd.read_pickle,
    # because pd.read_pickle may seek back to the beginning of the handle when it
    # falls back to pickle_compat. That is fatal for a concatenated pickle stream.
    from pandas.compat import pickle_compat
except Exception:
    pickle_compat = None

BYTES_PER_GB = 1024 ** 3

############################################################


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={value!r}; using {default}")
        return default


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={value!r}; using {default}")
        return default


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / BYTES_PER_GB


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / BYTES_PER_GB


def check_free_space(path: str | Path = "/") -> float:
    """Return free disk space in GB for the filesystem containing *path*."""
    disk_usage = psutil.disk_usage(str(path))
    return disk_usage.free / BYTES_PER_GB


def dataframe_mem_gb(df: pd.DataFrame) -> float:
    """Return the approximate deep memory footprint of a DataFrame in GB."""
    if df is None:
        return 0.0
    return float(df.memory_usage(deep=True).sum()) / BYTES_PER_GB


def _estimate_df_list_mem_gb(dfs: Iterable[pd.DataFrame]) -> float:
    return sum(dataframe_mem_gb(df) for df in dfs if df is not None)


def _target_chunk_ram_gb() -> float:
    """
    Choose a conservative target chunk size.

    The default is intentionally much smaller than available RAM because pandas
    concat needs temporary copies. Users can override via DGT_CHUNK_TARGET_RAM_GB.
    """
    explicit = os.environ.get("DGT_CHUNK_TARGET_RAM_GB")
    if explicit:
        return max(0.02, _env_float("DGT_CHUNK_TARGET_RAM_GB", 0.5))

    avail = available_ram_gb()
    return max(0.10, min(1.0, avail * 0.10))


def _min_free_ram_gb() -> float:
    return max(0.25, _env_float("DGT_MIN_FREE_RAM_GB", 2.0))


def _concat_ram_factor() -> float:
    return max(1.5, _env_float("DGT_CONCAT_RAM_FACTOR", 3.0))


def _require_ram(estimated_input_gb: float, operation: str) -> None:
    """
    Abort early if a pandas operation is likely to exceed available RAM.

    This is deliberately conservative: pandas concat, boolean masks and copies
    can temporarily require several times the apparent DataFrame size.
    """
    avail = available_ram_gb()
    min_free = _min_free_ram_gb()
    factor = _concat_ram_factor()
    required = estimated_input_gb * factor + min_free

    if avail < required:
        raise MemoryError(
            f"Not enough RAM for {operation}.\n"
            f"  Estimated input size: {estimated_input_gb:.2f} GB\n"
            f"  Available RAM:        {avail:.2f} GB\n"
            f"  Required RAM budget:  {required:.2f} GB "
            f"(factor={factor:.1f}, min_free={min_free:.1f} GB)\n"
            "Suggestions:\n"
            "  - fix T, width and/or tau values to reduce the grid before loading\n"
            "  - use a smaller model grid or fewer transitions\n"
            "  - increase RAM/swap\n"
            "  - lower DGT_CHUNK_TARGET_RAM_GB, e.g. export DGT_CHUNK_TARGET_RAM_GB=0.25\n"
            "  - lower DGT_MIN_FREE_RAM_GB only if you know what you are doing\n"
        )


###########################################################


def download_file(url, local_path, model_size_gb):
    """Download a model file if it is not present already."""
    local_path = Path(local_path)
    if local_path.exists():
        print(f"File {local_path} already exists. Skipping download.")
        return

    free_space_gb = check_free_space(local_path.parent if local_path.parent.exists() else ".")
    if free_space_gb < model_size_gb and "emissivities" in str(local_path):
        raise RuntimeError(
            f"Not enough disk space. Available: {free_space_gb:.2f} GB, "
            f"required: {model_size_gb} GB"
        )

    print(f"Downloading {url} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with local_path.open("wb") as file:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    file.write(block)
    print(f"Downloaded {local_path}")


###########################################################


def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)


############################################################


def _read_one_pickle_from_stream(handle):
    """
    Read exactly one object from a concatenated pickle stream.

    Do NOT use pandas.read_pickle(handle) here. With old pandas pickles and newer
    pandas versions, pandas.read_pickle may fall back to pandas' compatibility
    unpickler and seek the file handle back to the beginning. In a pickle stream
    that can make the loop read the first object forever, producing endless tmp
    chunks.
    """
    pos_before = handle.tell()

    try:
        obj = pickle.load(handle)
    except EOFError:
        raise
    except (AttributeError, ImportError, ModuleNotFoundError, TypeError):
        if pickle_compat is None:
            raise
        handle.seek(pos_before)
        obj = pickle_compat.Unpickler(handle).load()

    pos_after = handle.tell()
    if pos_after <= pos_before:
        raise RuntimeError(
            "Pickle stream reader did not advance. This would create an "
            "infinite read loop. Aborting before filling tmp/ with duplicates."
        )
    return obj


############################################################


def _chunk_sort_key(path):
    stem = Path(path).stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return stem


############################################################


def _prepare_grid_selection(transition, usertkin, userwidth, usertau):
    """Pre-compute the model columns/species needed for a specific user run."""
    userlines = [linename_obs2mdl(x) for x in transition]
    userspecies = [
        linename_obs2mdl(x)
        .replace("10", "")
        .replace("21", "")
        .replace("32", "")
        .replace("43", "")
        for x in userlines
    ]

    # Preserve order but avoid duplicates.
    userspecies = list(dict.fromkeys(userspecies))

    mdlcols = ["n_mean", "n_mean_mass", "tkin", "width", "fdense_thresh", "fdense_pl", "pl"]
    keepcols = userlines + mdlcols

    fixed_tau_constraints = []
    if isinstance(usertau, list) and len(usertau) > 0:
        for trans_tau in usertau:
            this_trans, this_tau = trans_tau.split("_")
            this_species = (
                linename_obs2mdl(this_trans)
                .replace("10", "")
                .replace("21", "")
                .replace("32", "")
                .replace("43", "")
            )
            fixed_tau_constraints.append(("tau_" + this_species, float(this_tau)))

    return {
        "userlines": userlines,
        "userspecies": userspecies,
        "keepcols": keepcols,
        "usertkin": usertkin,
        "userwidth": userwidth,
        "usertau": usertau,
        "fixed_tau_constraints": fixed_tau_constraints,
    }


def _select_relevant_columns(df: pd.DataFrame, selection: dict) -> pd.DataFrame:
    keepcols = selection["keepcols"]
    userspecies = selection["userspecies"]

    cols_to_keep = []
    missing_required = []

    for col in keepcols:
        if col in df.columns:
            cols_to_keep.append(col)
        else:
            missing_required.append(col)

    for col in df.columns:
        if col.startswith("tau_"):
            parts = col.split("_")
            if len(parts) >= 2 and parts[1] in userspecies:
                cols_to_keep.append(col)

    # Do not fail on every streamed object if not all columns exist in every object,
    # but the final concatenated table will fail later if required columns are absent.
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    if not cols_to_keep:
        return pd.DataFrame()

    return df.loc[:, cols_to_keep]


def _reduce_grid_chunk(df: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """
    Keep only columns and rows relevant for the current user input.

    Applying this reduction before writing temporary chunks is the main RAM/disk
    saver compared to the old approach, which wrote complete model chunks first.
    """
    if df.empty:
        return df

    df = _select_relevant_columns(df, selection)
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    usertkin = selection["usertkin"]
    userwidth = selection["userwidth"]

    if usertkin > 0:
        if "tkin" not in df.columns:
            raise KeyError("Column 'tkin' missing from model grid chunk")
        mask &= df["tkin"] == usertkin

    if userwidth > 0:
        if "width" not in df.columns:
            raise KeyError("Column 'width' missing from model grid chunk")
        mask &= df["width"] == userwidth

    for tau_col, tau_val in selection["fixed_tau_constraints"]:
        if tau_col not in df.columns:
            raise KeyError(f"Column {tau_col!r} missing from model grid chunk")
        mask &= df[tau_col] == tau_val

    reduced = df.loc[mask].copy()
    return reduced.reset_index(drop=True)


############################################################


def _write_reduced_chunk(chunk_objects, chunk_filename: Path, selection: dict) -> tuple[int, float]:
    """
    Concatenate raw streamed objects, reduce immediately, and write a small chunk.

    Returns
    -------
    rows_written, reduced_mem_gb
    """
    if not chunk_objects:
        return 0, 0.0

    input_mem_gb = _estimate_df_list_mem_gb(chunk_objects)
    _require_ram(input_mem_gb, operation="concatenating raw streamed model objects")

    if len(chunk_objects) == 1:
        raw = chunk_objects[0]
    else:
        raw = pd.concat(chunk_objects, ignore_index=True, copy=False)

    reduced = _reduce_grid_chunk(raw, selection)

    # Drop references before writing/returning.
    del raw
    chunk_objects.clear()
    gc.collect()

    if reduced.empty:
        return 0, 0.0

    reduced_mem_gb = dataframe_mem_gb(reduced)
    free_disk_gb = check_free_space(chunk_filename.parent)
    # Pickle size is not identical to RAM size; keep a conservative margin.
    if free_disk_gb < max(1.0, reduced_mem_gb * 2.0):
        raise RuntimeError(
            f"Not enough free disk space for temporary chunk {chunk_filename}.\n"
            f"  Free disk:          {free_disk_gb:.2f} GB\n"
            f"  Reduced chunk RAM:  {reduced_mem_gb:.2f} GB"
        )

    reduced.to_pickle(chunk_filename)
    rows = len(reduced)
    print(f"\n[INFO] Wrote reduced {chunk_filename} ({rows:,} rows, ~{reduced_mem_gb:.2f} GB RAM)")
    del reduced
    gc.collect()
    return rows, reduced_mem_gb


############################################################


def read_and_save_reduced_chunks(
    pklfile,
    selection: dict,
    chunk_prefix="reduced_chunk_",
    tmpdir="tmp",
    clean_tmp=True,
    max_pickle_objects=None,
):
    """
    Read a concatenated pickle stream and store reduced temporary chunks.

    The counter printed here is the number of serialized pickle objects read,
    not necessarily the number of physical (n,T,width) model groups.
    """
    pklfile = Path(pklfile)
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    if clean_tmp:
        for old_chunk in tmpdir.glob(f"{chunk_prefix}*.pkl"):
            old_chunk.unlink()

    total_size = pklfile.stat().st_size
    target_chunk_ram_gb = _target_chunk_ram_gb()
    max_pickle_objects = _env_int("DGT_MAX_PICKLE_OBJECTS", max_pickle_objects)

    print(
        "[INFO] RAM-aware pickle loading enabled: "
        f"available RAM={available_ram_gb():.2f}/{total_ram_gb():.2f} GB, "
        f"target raw chunk={target_chunk_ram_gb:.2f} GB, "
        f"min free RAM={_min_free_ram_gb():.2f} GB"
    )

    n_objects = 0
    total_rows = 0
    chunk_index = 0
    chunk_objects = []
    chunk_mem_gb = 0.0

    with pklfile.open("rb") as pk:
        while True:
            try:
                obj = _read_one_pickle_from_stream(pk)
            except EOFError:
                rows, _ = _write_reduced_chunk(
                    chunk_objects,
                    tmpdir / f"{chunk_prefix}{chunk_index}.pkl",
                    selection,
                )
                total_rows += rows
                print(
                    f"\n[INFO] Finished reading pickle stream: {n_objects:,} serialized objects; "
                    f"kept {total_rows:,} reduced rows"
                )
                break

            if not isinstance(obj, pd.DataFrame):
                obj = pd.DataFrame(obj)

            obj_mem_gb = dataframe_mem_gb(obj)
            chunk_objects.append(obj)
            chunk_mem_gb += obj_mem_gb
            n_objects += 1

            if max_pickle_objects is not None and n_objects > max_pickle_objects:
                raise RuntimeError(
                    f"Read more than {max_pickle_objects:,} pickle objects from {pklfile}. "
                    "This is suspicious; aborting to avoid filling disk/RAM."
                )

            if n_objects % 100 == 0:
                pos = pk.tell()
                pct = 100.0 * pos / total_size if total_size else 0.0
                print(
                    f"[INFO] Reading serialized pickle objects: {n_objects:,} "
                    f"({pos / BYTES_PER_GB:.2f}/{total_size / BYTES_PER_GB:.2f} GB, {pct:.1f}%; "
                    f"raw chunk ~{chunk_mem_gb:.2f} GB; free RAM {available_ram_gb():.2f} GB)",
                    end="\r",
                    flush=True,
                )

            # Flush early either when we hit the target or when available RAM is
            # getting close to the conservative budget.
            must_flush = chunk_mem_gb >= target_chunk_ram_gb
            low_ram = available_ram_gb() < (chunk_mem_gb * _concat_ram_factor() + _min_free_ram_gb())
            if must_flush or low_ram:
                rows, _ = _write_reduced_chunk(
                    chunk_objects,
                    tmpdir / f"{chunk_prefix}{chunk_index}.pkl",
                    selection,
                )
                total_rows += rows
                chunk_index += 1
                chunk_mem_gb = 0.0

    return {
        "objects_read": n_objects,
        "rows_kept": total_rows,
        "chunks_written": chunk_index + 1,
    }


############################################################


def load_pickle(file):
    return pd.read_pickle(file)


############################################################


def concatenate_reduced_chunks(chunk_prefix="reduced_chunk_", tmpdir="tmp"):
    """Concatenate already-reduced temporary chunks with RAM checks."""
    chunk_files = sorted(glob.glob(str(Path(tmpdir) / f"{chunk_prefix}*.pkl")), key=_chunk_sort_key)
    if not chunk_files:
        raise RuntimeError(f"No temporary chunks found in {tmpdir!r} with prefix {chunk_prefix!r}")

    print(f"[INFO] Concatenating {len(chunk_files)} reduced chunk file(s)")

    dfs = []
    loaded_mem_gb = 0.0
    for i, file in enumerate(chunk_files, start=1):
        df = pd.read_pickle(file)
        mem = dataframe_mem_gb(df)
        loaded_mem_gb += mem
        dfs.append(df)

        if i % 10 == 0 or i == len(chunk_files):
            print(
                f"[INFO] Loaded reduced chunk {i}/{len(chunk_files)} "
                f"(~{loaded_mem_gb:.2f} GB in DataFrames; free RAM {available_ram_gb():.2f} GB)"
            )

        _require_ram(loaded_mem_gb, operation="loading reduced model chunks")

    _require_ram(loaded_mem_gb, operation="final concatenation of reduced model chunks")
    grid = pd.concat(dfs, ignore_index=True, copy=False)
    del dfs
    gc.collect()

    return grid


############################################################


def read_stream_old(pklfile):
    """Legacy in-memory stream reader, now using safe pickle-stream loading."""
    objs = []
    with open(pklfile, "rb") as pk:
        n = 0
        while True:
            try:
                obj = _read_one_pickle_from_stream(pk)
            except EOFError:
                print()
                break
            objs.append(obj)
            n += 1
            print(f"[INFO] Reading serialized pickle objects: {n:,}", end="\r")
    return objs


############################################################


def _read_csv_reduced(gridfile, selection: dict, chunksize=250_000):
    """Read CSV grid in chunks and reduce immediately."""
    reduced_chunks = []
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(gridfile, chunksize=chunksize), start=1):
        reduced = _reduce_grid_chunk(chunk, selection)
        if not reduced.empty:
            reduced_chunks.append(reduced)
            total_rows += len(reduced)
        print(
            f"[INFO] CSV chunk {i}: kept {len(reduced):,} rows "
            f"(total kept {total_rows:,}; free RAM {available_ram_gb():.2f} GB)"
        )
        loaded_mem_gb = _estimate_df_list_mem_gb(reduced_chunks)
        _require_ram(loaded_mem_gb, operation="loading reduced CSV chunks")

    if not reduced_chunks:
        return pd.DataFrame()

    loaded_mem_gb = _estimate_df_list_mem_gb(reduced_chunks)
    _require_ram(loaded_mem_gb, operation="concatenating reduced CSV chunks")
    return pd.concat(reduced_chunks, ignore_index=True, copy=False)


############################################################


def read_grid_ndist(transition, usertkin, userwidth, usertau, powerlaw, type_of_models="std", usecsv=False):

    if not powerlaw:
        if usecsv:
            lratfile = "emissivities.csv"
        else:
            lratfile = "emissivities.pkl"
    else:
        if usecsv:
            lratfile = "emissivities_powerlaw.csv"
        else:
            lratfile = "emissivities_powerlaw.pkl"

    gridfile = "models_" + type_of_models + "/" + lratfile
    ensure_directory_exists("./models_" + type_of_models + "/")

    if type_of_models == "std43_incl_HCN_excl_C18O":
        model_size_gb = 40
    elif type_of_models == "co":
        model_size_gb = 9.2
    elif type_of_models == "thick":
        model_size_gb = 12
    else:
        model_size_gb = 33  # std

    urls = ["https://www.jpuschnig.com/dgt/" + gridfile]
    local_paths = [gridfile]

    for url, local_path in zip(urls, local_paths):
        download_file(url, local_path, model_size_gb)

    selection = _prepare_grid_selection(transition, usertkin, userwidth, usertau)

    print("[INFO] Down-select configuration")
    print("[INFO] Requested model lines:", selection["userlines"])
    print("[INFO] Requested species:", selection["userspecies"])
    if usertkin > 0:
        print(f"[INFO] Fixed Tkin: {usertkin}")
    if userwidth > 0:
        print(f"[INFO] Fixed width: {userwidth}")
    if isinstance(usertau, list) and len(usertau) > 0:
        print(f"[INFO] Fixed tau constraints: {selection['fixed_tau_constraints']}")
    else:
        print("[INFO] Tau is free; keeping tau columns for requested species")

    if usecsv:
        print("[INFO] Reading models from CSV in RAM-aware chunks")
        grid = _read_csv_reduced(gridfile, selection)
    else:
        tmpdir = os.environ.get("DGT_TMPDIR", "tmp")
        read_and_save_reduced_chunks(gridfile, selection, tmpdir=tmpdir)
        print("[INFO] Concatenating reduced model chunks")
        grid = concatenate_reduced_chunks(tmpdir=tmpdir)

    print("[INFO] Final model reduction")

    if grid.empty:
        raise RuntimeError(
            "No model rows remain after applying the requested T/width/tau/line selection. "
            "Please check example.py and the model-grid configuration."
        )

    before = len(grid)
    grid = grid.drop_duplicates().reset_index(drop=True)
    after = len(grid)
    if after != before:
        print(f"[INFO] Removed duplicate model rows: {before:,} -> {after:,}")

    required_columns = ["n_mean", "n_mean_mass", "tkin", "width"] + selection["userlines"]
    missing = [col for col in required_columns if col not in grid.columns]
    if missing:
        raise KeyError(
            "Required model columns are missing after loading/down-selection: "
            + ", ".join(missing)
        )

    grid["n_mean"] = np.log10(grid["n_mean"])
    grid["n_mean_mass"] = np.log10(grid["n_mean_mass"])

    print(
        f"[INFO] Final grid size: {len(grid):,} rows, {len(grid.columns):,} columns, "
        f"~{dataframe_mem_gb(grid):.2f} GB RAM"
    )

    return grid


###########################################################################
###########################################################################


def linename_obs2mdl(line: str):

    ll = line.lower()
    if ll == "co10":
        ll = "12co10"
    elif ll == "co21":
        ll = "12co21"
    elif ll == "co32":
        ll = "12co32"
    elif ll == "co43":
        ll = "12co43"

    return str(ll)


###########################################################################
###########################################################################


def linename_mdl2obs(line: str):

    ll = line.upper()
    if ll == "12CO10":
        ll = "CO10"
    elif ll == "12CO21":
        ll = "CO21"
    elif ll == "12CO32":
        ll = "CO32"
    elif ll == "12CO43":
        ll = "CO43"

    return str(ll)
