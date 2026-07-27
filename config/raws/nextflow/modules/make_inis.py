#!/usr/bin/env python3
import argparse
import csv
import os
from configparser import ConfigParser

parser = argparse.ArgumentParser(
    description="Generate every RAWS sub-basin ini from a template ini.")
parser.add_argument("--template-ini", required=True)
parser.add_argument("--domains", required=True, help="M<id>_domains.csv manifest")
parser.add_argument("--outdir", required=True)
parser.add_argument("--upstream-timeout", default="600")
args = parser.parse_args()

with open(args.domains, newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["name"]]

clone_dir = os.path.dirname(os.path.abspath(args.domains))
outdir = os.path.abspath(args.outdir)

for row in rows:
    name = row["name"]
    inflow = row["upstream_names"].split(",") if row["upstream_names"] else []
    mask = os.path.normpath(os.path.join(clone_dir, row["mask_map"]))

    ini = ConfigParser(interpolation=None)  # relativeElevationFiles holds a %04d
    ini.optionxform = str                   # the model reads mixed-case keys
    ini.read(args.template_ini)

    ini["globalOptions"]["outputDir"] = os.path.join(outdir, name)
    ini["globalOptions"]["cloneMap"] = os.path.normpath(
        os.path.join(clone_dir, row["clone_map"]))
    ini["globalOptions"]["landmask"] = mask
    ini["reportingOptions"]["landmask_for_reporting"] = mask
    ini["routingOptions"]["upstream_discharge_output_file"] = os.path.join(
        outdir, name, "%s.bin" % name)
    ini["routingOptions"]["upstream_discharge_input_files"] = ",".join(
        os.path.join(outdir, up, "%s.bin" % up) for up in inflow) or "None"
    ini["routingOptions"]["upstream_discharge_timeout"] = args.upstream_timeout

    with open("%s.ini" % name, "w") as handle:
        ini.write(handle)