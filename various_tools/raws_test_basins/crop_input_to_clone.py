#!/usr/bin/env python
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

try:                                    # six.moves keeps py2/py3 parity with the model
    from six.moves.configparser import RawConfigParser
except ImportError:                     # pragma: no cover - py3 fallback
    from configparser import RawConfigParser


NETCDF_EXTS = (".nc", ".nc4")
PCRMAP_EXTS = (".map", ".ldd")
COPY_EXTS = (".tbl", ".txt", ".asc", ".dat")     # copied verbatim, not cropped
KNOWN_EXTS = NETCDF_EXTS + PCRMAP_EXTS + COPY_EXTS

# Reporting keys hold comma-separated *variable names*, never file paths.
REPORTING_VARLIST_KEYS = {
    "outDailyTotNC", "outMonthTotNC", "outMonthAvgNC", "outMonthEndNC",
    "outAnnuaTotNC", "outAnnuaAvgNC", "outAnnuaEndNC",
}


def run(cmd, dry_run):
    """Run a subprocess (or print it under --dry-run)."""
    cmd = [str(c) for c in cmd]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def resolve_tool(name):
    """Invocable path for a CLI tool, preferring PATH then the pixi env."""
    found = shutil.which(name)
    if found:
        return found
    repo_root = Path(__file__).resolve().parents[2]   # <repo>/various_tools/raws_test_basins/
    candidate = repo_root / ".pixi" / "envs" / "default" / "bin" / name
    if candidate.exists():
        return str(candidate)
    sys.exit(f"ERROR: required tool '{name}' not found on PATH or in the pixi env.\n"
             f"       Launch this script under 'pixi run' from the model root.")


# ---------------------------------------------------------------------------
# clone extent
# ---------------------------------------------------------------------------

Clone = namedtuple("Clone", "rows cols cellsize xmin xmax ymin ymax")


def read_clone(clone_path, mapattr):
    """Parse a clone .map's geometry (mirrors virtualOS.getMapAttributesALL)."""
    out = subprocess.run([mapattr, "-p", str(clone_path)],
                         capture_output=True, text=True, check=True).stdout
    attr = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            attr[parts[0]] = parts[1]
    rows, cols = int(float(attr["rows"])), int(float(attr["columns"]))
    cs, x, y = float(attr["cell_length"]), float(attr["xUL"]), float(attr["yUL"])
    return Clone(rows, cols, cs, x, x + cols * cs, y - rows * cs, y)


def write_clone_grid(clone, path):
    """Write a CDO grid description matching the clone's EXACT grid.

    Latitude is emitted north-to-south (yinc < 0): the model derives the input
    cellsize from lat[0]-lat[1] and expects that to be positive. Because the grid
    has the clone's exact size and origin, every netCDF remapped onto it lands
    factor-1 on the clone -- the model performs no runtime regridding, and the
    read-time crop can never overshoot the clone's row/column count.
    """
    xfirst = clone.xmin + 0.5 * clone.cellsize
    yfirst = clone.ymax - 0.5 * clone.cellsize     # first row = northernmost centre
    Path(path).write_text(
        "gridtype = lonlat\n"
        f"xsize = {clone.cols}\n"
        f"ysize = {clone.rows}\n"
        f"xfirst = {xfirst!r}\n"
        f"xinc = {clone.cellsize!r}\n"
        f"yfirst = {yfirst!r}\n"
        f"yinc = {-clone.cellsize!r}\n"
    )


# ---------------------------------------------------------------------------
# ini parsing + path resolution (replicates model/virtualOS.getFullPath)
# ---------------------------------------------------------------------------

def is_absolute(path):
    """Match getFullPath's absolute test: leading '/', 'X:' drive, or http URL."""
    return path.startswith("/") or (len(path) > 1 and path[1] == ":") or path.startswith("http")


def get_full_path(input_path, input_dir):
    """Resolve a raw ini value against inputDir (relative) or leave it (absolute)."""
    if is_absolute(input_path):
        return input_path
    return input_dir.rstrip("/") + "/" + input_path


def looks_like_path(value):
    """The model's file-vs-constant heuristic (singleTryReadPCRmapClone)."""
    if value in ("", "None", "False", "True", "Maximum", "Input"):
        return False
    # purely numeric (digits/dot/minus) -> a scalar constant, not a file
    return not re.match(r"^[0-9.\-]*$", value)


def collect_input_files(config, input_dir):
    """{resolved_source_path: (raw_value, is_absolute)} for every input file, de-duped."""
    files = {}

    def add(raw_value):
        raw_value = raw_value.strip()
        files.setdefault(get_full_path(raw_value, input_dir),
                         (raw_value, is_absolute(raw_value)))

    for section in config.sections():
        for key, value in config.items(section):
            value = (value or "").strip()

            # reporting variable-name lists and the landcover-type list are never files
            if key in REPORTING_VARLIST_KEYS or key == "landCoverTypes":
                continue

            # relativeElevationFiles is a %04d template expanded over the levels list;
            # substitute on the raw (relative) template so add() classifies it correctly.
            if key == "relativeElevationFiles" and "%" in value:
                levels = config.get(section, "relativeElevationLevels", fallback="")
                for lvl in (x for x in levels.split(",") if x.strip()):
                    add(value % (float(lvl) * 100))
            elif looks_like_path(value) and value.lower().endswith(KNOWN_EXTS):
                add(value)

    return files


def read_time_window(config):
    """(startTime, endTime) strings from the ini, or None if not both present."""
    for section in config.sections():
        if config.has_option(section, "startTime") and config.has_option(section, "endTime"):
            return (config.get(section, "startTime").strip(),
                    config.get(section, "endTime").strip())
    return None


# ---------------------------------------------------------------------------
# per-file cropping
# ---------------------------------------------------------------------------

# Invariant context for a crop batch — built once, passed to every worker.
CropCtx = namedtuple("CropCtx",
                     "clone input_dir output_dir tools dry_run grid_file time_start time_end")


def output_path_for(src_resolved, is_abs, input_dir, output_dir):
    """Destination for a source path under output_dir.

    Relative inputs keep their path relative to input_dir. Absolute inputs (meteo
    forcing here) are flattened to a clean ``forcing/<basename>`` so the source
    filesystem path -- including its owner's scratch dir -- is never reproduced in
    the output tree or the generated ini.
    """
    if is_abs:
        rel = Path("forcing") / Path(src_resolved).name
    else:
        rel = Path(os.path.relpath(src_resolved, input_dir))
    return Path(output_dir) / rel


def detect_nc_dims(src, ncks):
    """Return (lat_dim, lon_dim) names for a netCDF file."""
    cdl = subprocess.run([ncks, "--cdl", "-m", str(src)],
                         capture_output=True, text=True, check=True).stdout
    dims = set(re.findall(r"^\s*(\w+)\s*=\s*\d+\s*;", cdl, re.MULTILINE))
    lat = next((d for d in ("lat", "latitude", "y") if d in dims), None)
    lon = next((d for d in ("lon", "longitude", "x") if d in dims), None)
    return lat, lon


def nc_time_years(src, cdo):
    """Distinct calendar years on the file's time axis (sorted ints), [] if none.

    Distinguishes a genuine multi-year series (forcing, annual land cover) -- which
    we trim to the run window -- from a 12-month climatology (a single year) or a
    static map (no time axis), which are remapped whole and left untouched.
    """
    r = subprocess.run([cdo, "-s", "showyear", str(src)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return sorted({int(y) for y in r.stdout.split()})


def crop_file(src, dst, ctx):
    """Crop one input to the clone.

    .nc  -> cdo remapnn onto the clone's EXACT grid. Nearest-neighbour is
            piecewise-constant: lossless for same-resolution inputs (it re-picks
            the aligned cell) and, for coarser inputs, it block-replicates the
            coarse value -- exactly the model's own downscaling. The result lands
            factor-1 on the clone, so the model does NO runtime regridding and the
            read-time crop cannot overshoot the clone's row/column count. CDO
            preserves the time dimension, variable names, and _FillValue.
    .map -> gdal_translate -projwin: a pure window crop to the clone's exact bbox.
            The clone aligns to the global grid, so this is an exact cell cut-out
            with NO resampling/regridding -- categorical maps such as the LDD keep
            their 1-9 flow directions and value scale. (resample is avoided: it
            refuses ldd maps and would interpolate other types.)
    else -> copied verbatim.
    """
    low = str(src).lower()
    if low.endswith(NETCDF_EXTS):
        lat, lon = detect_nc_dims(src, ctx.tools["ncks"])
        if not (lat and lon):
            print(f"  WARN no lat/lon dims, copying unchanged: {src}")
            if not ctx.dry_run:
                shutil.copy2(src, dst)
            return
        if not ctx.dry_run and dst.exists():
            dst.unlink()                             # cdo won't overwrite in place
        # CDO chain runs right-to-left: subset the time axis to the run window
        # FIRST (cheap -- reads only the needed timesteps), then remapnn onto the
        # clone. Only a genuine multi-year series that overlaps the window is
        # trimmed; a 12-month climatology or a static map is remapped whole.
        chain = [f"remapnn,{ctx.grid_file}"]
        if ctx.time_start:
            years = nc_time_years(src, ctx.tools["cdo"])
            if len(years) > 1 and min(years) <= int(ctx.time_end[:4]) \
                              and max(years) >= int(ctx.time_start[:4]):
                chain.append(f"-seldate,{ctx.time_start}T00:00:00,{ctx.time_end}T23:59:59")
        run([ctx.tools["cdo"], "-s", *chain, src, dst], ctx.dry_run)
    elif low.endswith(PCRMAP_EXTS):
        if not ctx.dry_run and dst.exists():
            dst.unlink()                             # gdal_translate won't overwrite
        run([ctx.tools["gdal_translate"], "-q", "-of", "PCRaster", "-projwin",
             ctx.clone.xmin, ctx.clone.ymax, ctx.clone.xmax, ctx.clone.ymin, src, dst],
            ctx.dry_run)
    elif not ctx.dry_run:
        shutil.copy2(src, dst)


def _crop_one(src_resolved, is_abs, ctx):
    """Worker: crop one existing file. Returns (src_resolved, err_tail); err_tail is None on success."""
    dst = output_path_for(src_resolved, is_abs, ctx.input_dir, ctx.output_dir)
    if not ctx.dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)   # exist_ok -> thread-safe
    try:
        crop_file(Path(src_resolved), dst, ctx)
        return (src_resolved, None)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()
        return (src_resolved, tail[-1] if tail else "(no stderr)")


def crop_all(files, ctx, jobs):
    """Crop/copy every source file, up to `jobs` in parallel.

    Returns (n_done, n_missing, n_failed). Each crop is an independent subprocess,
    so a thread pool parallelizes them without GIL contention. Missing files and
    all tallying/printing happen in the main thread to keep output clean.
    """
    present, n_missing = [], 0
    for src_resolved, (_, is_abs) in sorted(files.items()):
        if Path(src_resolved).is_file():
            present.append((src_resolved, is_abs))
        else:
            print(f"  MISS {src_resolved}")
            n_missing += 1

    # dry-run stays serial so the planned-command lines print in a stable order.
    if ctx.dry_run or jobs <= 1:
        results = [_crop_one(sr, ia, ctx) for sr, ia in present]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_crop_one, sr, ia, ctx) for sr, ia in present]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    n_done = n_failed = 0
    for src_resolved, tail in results:
        if tail is None:
            n_done += 1
        else:
            print(f"  FAIL {src_resolved}\n       {tail}")
            n_failed += 1
    return n_done, n_missing, n_failed


# ---------------------------------------------------------------------------
# cropped ini
# ---------------------------------------------------------------------------

def _sub_key(text, key, new_value):
    """Replace 'key = ...' (first occurrence, preserving leading whitespace)."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    return pattern.sub(lambda m: m.group(1) + new_value, text, count=1)


def _sub_value(text, old_value, new_value):
    """Repoint any 'key = <old_value>' line to new_value (anchored to a full RHS).

    Unlike a blind str.replace, this only rewrites a value that is the entire
    right-hand side of an assignment, so it never touches comments or a path that
    merely contains old_value as a substring.
    """
    pattern = re.compile(rf"^(\s*\S[^=\n]*=\s*){re.escape(old_value)}\s*$", re.MULTILINE)
    return pattern.sub(lambda m: m.group(1) + new_value, text)


def write_cropped_ini(ini_path, output_dir, clone_dst, mask_dst, files, input_dir):
    """Emit a ready-to-run ini: repoint inputDir, clone, landmask, forcing inputs."""
    text = Path(ini_path).read_text()
    text = _sub_key(text, "inputDir", str(output_dir))
    text = _sub_key(text, "cloneMap", str(clone_dst))
    if mask_dst is not None:
        text = _sub_key(text, "landmask", str(mask_dst))
        text = _sub_key(text, "landmask_for_reporting", str(mask_dst))

    # Repoint absolute-path inputs (forcing) to their flattened forcing/ location.
    for src_resolved, (raw_value, is_abs) in files.items():
        if is_abs:
            mirrored = output_path_for(src_resolved, True, input_dir, output_dir)
            text = _sub_value(text, raw_value, str(mirrored))

    out_ini = Path(output_dir) / (Path(ini_path).stem + "_cropped.ini")
    out_ini.write_text(text)
    print(f"  cropped ini -> {out_ini}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Crop PCR-GLOBWB input data to a clone domain.")
    ap.add_argument("--ini", required=True, help="model .ini listing the input files")
    ap.add_argument("--clone", required=True, help="target clone .map (defines the extent)")
    ap.add_argument("--output", required=True, help="output directory for the cropped tree")
    ap.add_argument("--mask", default=None,
                    help="landmask .map for the cropped ini "
                         "(default: sibling <clone-stem>.mask.map if present)")
    ap.add_argument("--buffer-cells", type=int, default=5,
                    help="deprecated/ignored: netCDF is now remapped onto the clone's "
                         "exact grid (no buffer needed); kept for CLI compatibility")
    ap.add_argument("--full-time", action="store_true",
                    help="keep the full time axis; do NOT trim multi-year netCDF to the "
                         "ini's startTime..endTime (default: trim to the run window)")
    ap.add_argument("--jobs", "-j", type=int, default=min(8, os.cpu_count() or 4),
                    help="number of files to crop in parallel (default min(8, ncpu))")
    ap.add_argument("--no-ini", action="store_true", help="do not write a cropped ini")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned actions without writing anything")
    args = ap.parse_args(argv)

    ini_path = Path(args.ini).resolve()
    clone_path = Path(args.clone).resolve()
    output_dir = Path(args.output).resolve()
    for label, path in (("ini", ini_path), ("clone", clone_path)):
        if not path.is_file():
            sys.exit(f"ERROR: {label} not found: {path}")

    tools = {name: resolve_tool(name) for name in ("mapattr", "gdal_translate", "ncks", "cdo")}

    clone = read_clone(clone_path, tools["mapattr"])
    print(f"Clone {clone_path.name}: {clone.rows}x{clone.cols} @ {clone.cellsize:.6g}deg  "
          f"bbox lon[{clone.xmin:.4f},{clone.xmax:.4f}] lat[{clone.ymin:.4f},{clone.ymax:.4f}]")

    # CDO grid description for the clone; every netCDF is remapped onto it.
    grid_file = output_dir / "clone_grid.txt"

    config = RawConfigParser()
    config.optionxform = str            # preserve camelCase keys, no % interpolation
    config.read(ini_path)
    input_dir = config.get("globalOptions", "inputDir").strip()
    print(f"inputDir = {input_dir}")

    # Temporal window: trim genuine multi-year netCDF (forcing, annual series) to
    # the ini's run period so we don't remap decades when only a few years are used.
    time_start, time_end = (None, None)
    if not args.full_time:
        win = read_time_window(config)
        if win:
            time_start, time_end = win
            print(f"time window (from ini): {time_start} .. {time_end}  "
                  f"(multi-year netCDF trimmed to this range)")
        else:
            print("no startTime/endTime found in ini; keeping full time axis")

    files = collect_input_files(config, input_dir)
    jobs = 1 if args.dry_run else max(1, args.jobs)
    print(f"Resolved {len(files)} unique input files ({jobs} parallel job(s)).\n")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_clone_grid(clone, grid_file)          # must exist before the workers run

    ctx = CropCtx(clone, input_dir, output_dir, tools, args.dry_run, grid_file,
                  time_start, time_end)
    n_done, n_missing, n_failed = crop_all(files, ctx, jobs)

    # Copy clone + mask into the output tree for the cropped ini.
    clone_dst = output_dir / "clone_maps" / clone_path.name
    mask_src = Path(args.mask) if args.mask else \
        clone_path.with_name(clone_path.name.replace(".clone.map", ".mask.map"))
    mask_dst = output_dir / "clone_maps" / mask_src.name if mask_src.is_file() else None
    if not mask_dst:
        print(f"  WARN landmask not found ({mask_src}); cropped ini landmask left unchanged")

    if not args.dry_run:
        clone_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clone_path, clone_dst)
        if mask_dst:
            shutil.copy2(mask_src, mask_dst)
        if not args.no_ini:
            write_cropped_ini(ini_path, output_dir, clone_dst, mask_dst, files, input_dir)

    print(f"\nDone: {n_done} cropped/copied, {n_missing} missing, {n_failed} failed.")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
