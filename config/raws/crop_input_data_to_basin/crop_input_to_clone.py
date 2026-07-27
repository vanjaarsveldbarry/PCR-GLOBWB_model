#!/usr/bin/env python
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from configparser import RawConfigParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))
from virtualOS import getMapAttributesALL

NETCDF_EXTS = (".nc", ".nc4")
PCRMAP_EXTS = (".map", ".ldd")
COPY_EXTS = (".tbl", ".txt", ".asc", ".dat")
KNOWN_EXTS = NETCDF_EXTS + PCRMAP_EXTS + COPY_EXTS
CLONE_KEYS = {"cloneMap", "landmask", "landmask_for_reporting"}


def read_clone(clone_path):
    Clone = namedtuple("Clone", "rows cols cellsize xmin xmax ymin ymax")
    attr = getMapAttributesALL(str(clone_path))
    rows, cols, cs = int(attr["rows"]), int(attr["cols"]), attr["cellsize"]
    x, y = attr["xUL"], attr["yUL"]
    return Clone(rows, cols, cs, x, x + cols * cs, y - rows * cs, y)


def write_clone_grid(clone, path):
    # lat runs north-to-south (yinc < 0) so the model reads a positive cellsize
    Path(path).write_text(
        "gridtype = lonlat\n"
        f"xsize = {clone.cols}\n"
        f"ysize = {clone.rows}\n"
        f"xfirst = {clone.xmin + 0.5 * clone.cellsize!r}\n"
        f"xinc = {clone.cellsize!r}\n"
        f"yfirst = {clone.ymax - 0.5 * clone.cellsize!r}\n"
        f"yinc = {-clone.cellsize!r}\n"
    )


def collect_input_files(config, input_dir):
    files = {}   # {resolved_path: (raw_value, is_abs)}

    def add(raw):
        raw = raw.strip()
        abs_ = raw.startswith("/") or (len(raw) > 1 and raw[1] == ":")
        resolved = raw if abs_ else input_dir.rstrip("/") + "/" + raw
        files.setdefault(resolved, (raw, abs_))

    for section in config.sections():
        for key, value in config.items(section):
            value = (value or "").strip()
            if key in CLONE_KEYS:
                continue
            if key == "relativeElevationFiles" and "%" in value:
                levels = config.get(section, "relativeElevationLevels", fallback="")
                for lvl in filter(str.strip, levels.split(",")):
                    add(value % (float(lvl) * 100))
            elif value.lower().endswith(KNOWN_EXTS):
                add(value)
    return files


def output_path(resolved, is_abs, input_dir, output_dir):
    # absolute inputs (forcing) are flattened to forcing/<name>, hiding the source path
    rel = Path("forcing") / Path(resolved).name if is_abs \
        else Path(os.path.relpath(resolved, input_dir))
    return Path(output_dir) / rel


def write_cropped_ini(ini_path, output_dir, clone_dst, mask_dst, files, input_dir):
    def sub_key(text, key, value):
        return re.sub(rf"^(\s*{re.escape(key)}\s*=\s*).*$",
                      lambda m: m.group(1) + value, text, count=1, flags=re.MULTILINE)

    text = Path(ini_path).read_text()
    text = sub_key(text, "inputDir", str(output_dir))
    text = sub_key(text, "cloneMap", str(clone_dst))
    text = sub_key(text, "landmask", str(mask_dst))
    text = sub_key(text, "landmask_for_reporting", str(mask_dst))
    for resolved, (raw, is_abs) in files.items():
        if is_abs:
            mirrored = str(output_path(resolved, True, input_dir, output_dir))
            text = re.sub(rf"^(\s*\S[^=\n]*=\s*){re.escape(raw)}\s*$",
                          lambda m: m.group(1) + mirrored, text, flags=re.MULTILINE)
    out_ini = Path(output_dir) / (Path(ini_path).stem + "_cropped.ini")
    out_ini.write_text(text)
    print(f"  cropped ini -> {out_ini}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crop PCR-GLOBWB input data to a clone domain.")
    ap.add_argument("--ini", required=True, help="model .ini listing the input files")
    ap.add_argument("--clone", required=True, help="target clone .map (defines the extent)")
    ap.add_argument("--output", required=True, help="output directory for the cropped tree")
    args = ap.parse_args()

    ini_path = Path(args.ini).resolve()
    clone_path = Path(args.clone).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_file = output_dir / "clone_grid.txt"

    config = RawConfigParser()
    config.optionxform = str            # preserve camelCase keys, no % interpolation
    config.read(ini_path)
    input_dir = config.get("globalOptions", "inputDir").strip()
    window = (config.get("globalOptions", "startTime").strip(),
              config.get("globalOptions", "endTime").strip())
    files = collect_input_files(config, input_dir)

    clone = read_clone(clone_path)
    write_clone_grid(clone, grid_file)

    def crop_file(item):
        src, (_, is_abs) = item
        src = Path(src)
        dst = output_path(src, is_abs, input_dir, output_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        low = str(src).lower()
        if low.endswith(NETCDF_EXTS):
            if dst.exists():
                dst.unlink()                             # cdo won't overwrite in place
            # remapnn onto the clone's exact grid -> model does no runtime regridding
            chain = [f"remapnn,{grid_file}"]
            start, end = window
            if start:
                r = subprocess.run(["cdo", "-s", "showyear", str(src)],
                                   capture_output=True, text=True)
                years = sorted({int(y) for y in r.stdout.split()}) if r.returncode == 0 else []
                if len(years) > 1 and min(years) <= int(end[:4]) and max(years) >= int(start[:4]):
                    chain.append(f"-seldate,{start}T00:00:00,{end}T23:59:59")
            subprocess.run(["cdo", "-s", *chain, str(src), str(dst)], check=True,
                           capture_output=True, text=True)
        elif low.endswith(PCRMAP_EXTS):
            if dst.exists():
                dst.unlink()                             # gdal_translate won't overwrite
            # projwin is an exact cell cut-out (no resampling) -> keeps ldd flow directions
            subprocess.run([str(c) for c in ("gdal_translate", "-q", "-of", "PCRaster",
                           "-projwin", clone.xmin, clone.ymax, clone.xmax, clone.ymin, src, dst)],
                           check=True, capture_output=True, text=True)
        else:
            shutil.copy2(src, dst)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(crop_file, sorted(files.items())))

    clone_dst = output_dir / "clone_maps" / clone_path.name
    mask_src = clone_path.with_name(clone_path.name.replace(".clone.map", ".mask.map"))
    mask_dst = output_dir / "clone_maps" / mask_src.name
    clone_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clone_path, clone_dst)
    shutil.copy2(mask_src, mask_dst)
    write_cropped_ini(ini_path, output_dir, clone_dst, mask_dst, files, input_dir)

    print(f"\nDone: {len(files)} cropped/copied.")
