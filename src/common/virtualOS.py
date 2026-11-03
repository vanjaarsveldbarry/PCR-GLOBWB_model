import calendar
import datetime
import functools
import inspect
import logging
import math
import re
import struct

import netCDF4 as nc
import numpy as np
import pcraster as pcr
import pyinterp.backends.xarray
import xarray as xr
import zarr
from osgeo import gdal

from common import utils

gdal.UseExceptions()

logger = logging.getLogger(__name__)

# file cache to reduce opening and closing files
filecache = dict()

# global variables
MV = 1e20
smallNumber = 1e-39

pi = math.pi

# netCDF file suffixes that can be used
netcdf_suffixes = (".nc4", ".nc", ".nc3")

# maximum number of tries for reading files
max_num_of_tries = 6

# map PCR-GLOBWB variable names to the names in the netCDF file
netcdf_variable_aliases = {
    "evapotranspiration": "referencePotET",
    "kc": "Cropcoefficient",
    "interceptCapInput": "Interceptioncapacity",
    "coverFractionInput": "Coverfraction",
    "fracVegCover": "vegetation_fraction",
    "minSoilDepthFrac": "minRootDepthFraction",
    "maxSoilDepthFrac": "maxRootDepthFraction",
    "arnoBeta": "arnoSchemeBeta",
}


def retry(func):
    # retries transient failures when reading over OpenDAP
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        source = next(iter(arguments.values()))
        absolute_path = arguments.get("absolutePath")
        if not any(str(p).startswith("http") for p in (source, absolute_path)):
            return func(*args, **kwargs)
        for attempt in range(1, max_num_of_tries + 1):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt == max_num_of_tries:
                    logger.error(f"CANNOT READ file/value: {source}")
                    raise
                logger.warning(f"Re-try to read file/value: {source}")

    return wrapper


def readUpstreamDischarge(
    ncFile,
    varName="automatic",
    dateInput=None,
    useDoy=None,
    LatitudeLongitude=True,
    specificFillValue=None,
):
    logger.debug(f"Reading Upstream Discharge: {ncFile}")
    lon = pcr.pcr2numpy(pcr.xcoordinate(pcr.boolean(1.0)), np.nan)[0, :]
    lat = np.sort(pcr.pcr2numpy(pcr.ycoordinate(pcr.boolean(1.0)), np.nan)[:, 0])

    ds = xr.open_dataset(ncFile, chunks="auto", engine="netcdf4")
    ds = ds.sel(time=dateInput).compute()
    ds = ds.reindex(lat=lat, lon=lon).sortby("lat", ascending=False)
    ds = ds.fillna(0.0)
    cropData = ds.discharge.values
    outPCR = pcr.numpy2pcr(
        pcr.Scalar, regridData2FinerGrid(1, cropData, float(0)), float(0)
    )
    ds.close()
    return outPCR


def checkVariableInNC(ncFile, varName):

    logger.debug(f"Checking whether {varName} is defined in {ncFile}")

    if ncFile in list(filecache.keys()):
        f = filecache[ncFile]

    else:
        f = nc.Dataset(ncFile)
        filecache[ncFile] = f

    varName = str(varName)

    return varName in list(f.variables.keys())


def writePCRmapToDir(v, outFileName, outDir):
    # v: input map file name or value; if the input map and the clone have different
    # grids, the map is resampled
    fullFileName = utils.getFullPath(outFileName, outDir)
    logger.debug(f"Writing a pcraster map to : {fullFileName}")
    pcr.report(v, fullFileName)


@retry
def readPCRmapClone(
    v,
    varName="automatic",
    dateInput=None,
    useDoy=None,
    LatitudeLongitude=True,
    specificFillValue=None,
    absolutePath=None,
    isLddMap=False,
    cover=None,
    isNomMap=False,
    bilinear=False,
):
    # v: input map file name or value; if the input map and the clone have different
    # grids, the map is resampled
    # varName, dateInput, useDoy, LatitudeLongitude and specificFillValue require
    # netCDF or zarr input; bilinear requires netCDF input

    def readNetcdf(ncFile, varName):

        # EHS (19 Apr 2013): convert a netCDF (tss) file to a PCRaster map with clone checking;
        # only works for square cells and if cellsizeClone <= cellsizeInput

        if varName != "automatic":
            logger.debug(f"reading variable: {varName} from the file: {ncFile}")

        if ncFile in list(filecache.keys()):
            f = filecache[ncFile]

        else:
            f = nc.Dataset(ncFile)
            filecache[ncFile] = f

        varName = str(varName)

        if LatitudeLongitude:
            try:
                f.variables["lat"] = f.variables["latitude"]
                f.variables["lon"] = f.variables["longitude"]
            except Exception:
                pass

        if varName == "automatic":
            nc_dims = [dim for dim in f.dimensions]
            nc_vars = [var for var in f.variables]
            for var in nc_vars:
                if var not in nc_dims and var not in [
                    "lat",
                    "lon",
                    "latitude",
                    "longitude",
                ]:
                    varName = var
            logger.debug(f"reading variable: {varName} from the file: {ncFile}")

        alias = netcdf_variable_aliases.get(varName)
        if varName not in f.variables and alias in f.variables:
            f.variables[varName] = f.variables[alias]

        var = f.variables[varName]
        hasTime = "time" in var.dimensions

        if not hasTime and dateInput is not None:
            raise ValueError(f"{varName} in {ncFile} has no time dimension")

        if hasTime:
            if dateInput is None:
                logger.debug("Using the first time step in the netcdf file.")
                idx = 0
                if len(f.variables["time"]) > 1:
                    logger.warning(
                        "NOTE that there are more than one time steps in the netcdf file."
                    )

            else:

                date = dateInput
                if useDoy == "Yes":
                    logger.debug(
                        "Finding the date from climatology doy index (1 to 366)"
                    )
                    idx = int(dateInput) - 1
                # needed for netCDF files that contain only 12 monthly values (e.g. cropCoefficientWaterNC)
                elif useDoy == "month":
                    logger.debug(
                        "Finding the date from climatology month index (1 to 12)"
                    )
                    # make sure the date has the correct format
                    if isinstance(date, str):
                        date = datetime.datetime.strptime(str(date), "%Y-%m-%d")
                    idx = int(date.month) - 1
                else:
                    # make sure the date has the correct format
                    if isinstance(date, str):
                        date = datetime.datetime.strptime(str(date), "%Y-%m-%d")
                    date = datetime.datetime(date.year, date.month, date.day)
                    if useDoy == "yearly":
                        date = datetime.datetime(date.year, int(1), int(1))
                    if useDoy == "monthly":
                        date = datetime.datetime(date.year, date.month, int(1))
                    if (
                        useDoy == "yearly"
                        or useDoy == "monthly"
                        or useDoy == "daily_seasonal"
                        or useDoy == "daily"
                        or useDoy == "daily_per_monthly_file"
                    ):
                        # if the desired year is not available, use the first or last available year
                        first_year_in_nc_file = findFirstYearInNCTime(
                            f.variables["time"]
                        )
                        last_year_in_nc_file = findLastYearInNCTime(f.variables["time"])
                        if date.year < first_year_in_nc_file:
                            if (
                                date.day == 29
                                and date.month == 2
                                and calendar.isleap(date.year)
                                and not calendar.isleap(first_year_in_nc_file)
                            ):
                                date = datetime.datetime(
                                    first_year_in_nc_file, date.month, 28
                                )
                            else:
                                date = datetime.datetime(
                                    first_year_in_nc_file, date.month, date.day
                                )
                            logger.warning(
                                f"Date {dateInput} NOT available in {ncFile} (variable: {varName}); "
                                f"using {date.year}-{date.month}-{date.day} instead."
                            )
                        if date.year > last_year_in_nc_file:
                            if (
                                date.day == 29
                                and date.month == 2
                                and calendar.isleap(date.year)
                                and not calendar.isleap(last_year_in_nc_file)
                            ):
                                date = datetime.datetime(
                                    last_year_in_nc_file, date.month, 28
                                )
                            else:
                                date = datetime.datetime(
                                    last_year_in_nc_file, date.month, date.day
                                )
                            logger.warning(
                                f"Date {dateInput} NOT available in {ncFile} (variable: {varName}); "
                                f"using {date.year}-{date.month}-{date.day} instead."
                            )
                    try:
                        idx = nc.date2index(
                            date,
                            f.variables["time"],
                            calendar=f.variables["time"].calendar,
                            select="exact",
                        )
                        logger.debug(
                            f"{date:%Y-%m-%d} found in netcdf; using select='exact'"
                        )
                    except Exception:
                        logger.debug(
                            f"{date:%Y-%m-%d} not found; select='exact' not possible"
                        )
                        if useDoy == "daily":
                            idx = nc.date2index(
                                date,
                                f.variables["time"],
                                calendar=f.variables["time"].calendar,
                                select="after",
                            )
                            select = "after"
                        else:
                            try:
                                idx = nc.date2index(
                                    date,
                                    f.variables["time"],
                                    calendar=f.variables["time"].calendar,
                                    select="before",
                                )
                                select = "before"
                            except Exception:
                                idx = nc.date2index(
                                    date,
                                    f.variables["time"],
                                    calendar=f.variables["time"].calendar,
                                    select="after",
                                )
                                select = "after"
                        date_string = nc.num2date(
                            f.variables["time"][int(idx)],
                            f.variables["time"].units,
                            f.variables["time"].calendar,
                        )
                        logger.warning(
                            f"Date {date.year}-{date.month}-{date.day} 00:00:00 NOT available in {ncFile} "
                            f"(variable: {varName}); the '{select}' option is used, giving {date_string}."
                        )

            idx = int(idx)
            logger.debug(f"Using the date index {idx}")

            date_string = nc.num2date(
                f.variables["time"][int(idx)],
                f.variables["time"].units,
                f.variables["time"].calendar,
            )
            logger.debug(f"Using the datetime {date_string}")

        attributeInput = getNetcdfGridAttributes(f)
        cellsizeInput = attributeInput["cellsize"]
        sameClone = utils.isSameClone(attributeInput)

        factor = 1
        yslice = slice(None)
        xslice = slice(None)
        if not sameClone:
            factor, xslice, yslice = getCloneCropSlices(
                f.variables["lon"][:],
                f.variables["lat"][:],
                cellsizeInput,
                padding=1 if bilinear else 0,
            )

        # retrieve the data slice from the netCDF file
        if not hasTime:
            cropData = var[yslice, xslice]
        elif var.ndim == 4:
            logger.warning(
                "%s (%s) is 4D; last two dims read as lat/lon", ncFile, varName
            )
            # file with an additional layer
            cropData = var[idx, 0, yslice, xslice]
        else:
            # standard netCDF file
            cropData = var[idx, yslice, xslice]

        fillValue = specificFillValue
        if fillValue is None:
            fillValue = getattr(var, "_FillValue", getattr(var, "missing_value", MV))
        fillValue = float(fillValue)

        if bilinear and not sameClone:
            fineData = interpolateToClone(
                cropData, f.variables["lat"][yslice], f.variables["lon"][xslice]
            )
        else:
            fineData = regridData2FinerGrid(factor, cropData, fillValue)

        outPCR = pcr.numpy2pcr(pcr.Scalar, fineData, fillValue)

        if useDoy == "daily_per_monthly_file":
            # close the file on the last day of the month and remove it from the cache
            tomorrow = date + datetime.timedelta(days=1)
            if tomorrow.day == 1:
                f.close()
                del filecache[ncFile]

        return outPCR

    def readZarr(zarrFile, varName):
        if bilinear:
            raise ValueError(f"{zarrFile}: bilinear needs netCDF input")

        if dateInput is None or useDoy != "Yes":
            raise ValueError(f"{zarrFile} needs a 1-based dateInput and useDoy='Yes'")

        if zarrFile in filecache:
            f = filecache[zarrFile]
        else:
            f = zarr.convenience.open(zarrFile)
            filecache[zarrFile] = f

        keys = sorted(f.keys())

        yRef = "lat"
        xRef = "lon"
        if LatitudeLongitude:
            if "latitude" in keys:
                yRef = "latitude"
            if "longitude" in keys:
                xRef = "longitude"

        if varName == "automatic":
            dims = ["time", xRef, yRef]
            varName = [item for item in keys if item not in dims][0]
        var = f[varName]

        cellsizeInput = float(f[yRef][0] - f[yRef][1])
        factor, xslice, yslice = getCloneCropSlices(
            f[xRef][:], f[yRef][:], cellsizeInput
        )

        # stores are ordered (time, lon, lat); callers do arithmetic with the factors,
        # so NaN must become 0 rather than a missing value
        cropData = np.nan_to_num(var[int(dateInput) - 1, xslice, yslice]).T
        # numpy2pcr reads the raw buffer, so a transposed view would scramble the map
        cropData = cropData.copy(order="C")

        if specificFillValue is not None:
            fillValue = float(specificFillValue)
        elif var.fill_value is not None:
            fillValue = float(var.fill_value)
        else:
            fillValue = float(MV)

        return pcr.numpy2pcr(
            pcr.Scalar, regridData2FinerGrid(factor, cropData, fillValue), fillValue
        )

    def readPCRaster(mapFile):
        sameClone = utils.isSameClone(getPCRasterGridAttributes(mapFile))
        if sameClone:
            return pcr.readmap(mapFile)
        return gdalwarpPCR(mapFile, isLddMap or isNomMap)

    logger.debug(f"read file/value: {v}")

    this_is_a_netcdf_file = v.endswith(netcdf_suffixes)
    this_is_a_zarr_store = v.endswith(".zarr")
    if (
        not this_is_a_netcdf_file
        and not this_is_a_zarr_store
        and (
            varName != "automatic"
            or dateInput is not None
            or useDoy is not None
            or not LatitudeLongitude
            or specificFillValue is not None
            or bilinear
        )
    ):
        raise ValueError(
            f"{v} is not a netCDF or zarr file but netCDF/zarr options were given"
        )

    if v == "None":

        PCRmap = None

    elif not re.match(r"[0-9.-]*$", v):
        if absolutePath is not None:
            v = utils.getFullPath(v, absolutePath)

        if this_is_a_netcdf_file:

            logger.debug(f"read netcdf file: {v}")

            PCRmap = readNetcdf(v, varName)

        elif this_is_a_zarr_store:

            logger.debug(f"read zarr store: {v}")

            PCRmap = readZarr(v, varName)

        else:

            PCRmap = readPCRaster(v)
    else:
        PCRmap = pcr.spatial(pcr.scalar(float(v)))

    # make sure the values have the correct type
    if isLddMap:
        PCRmap = pcr.ifthen(pcr.scalar(PCRmap) < 10.0, PCRmap)
    if isLddMap:
        PCRmap = pcr.ifthen(pcr.scalar(PCRmap) > 0.0, PCRmap)
    if isLddMap:
        PCRmap = pcr.ldd(PCRmap)
    if isNomMap:
        PCRmap = pcr.ifthen(pcr.scalar(PCRmap) > 0.0, PCRmap)
    if isNomMap:
        PCRmap = pcr.nominal(PCRmap)

    if cover is not None:
        PCRmap = pcr.cover(PCRmap, cover)

    return PCRmap


def getCloneCropSlices(lon, lat, cellsizeInput, padding=0):
    attributeClone = utils.getCloneAttributes()
    cellsizeClone = attributeClone["cellsize"]

    factor = int(round(float(cellsizeInput) / float(cellsizeClone)))
    if factor > 1:
        logger.debug(f"Resampling cell size {cellsizeInput} to {cellsizeClone}")

    diffX = np.abs(lon - (attributeClone["xUL"] + 0.5 * cellsizeInput))
    xIdxSta = np.argmin(diffX)
    xIdxEnd = math.ceil(xIdxSta + attributeClone["cols"] / factor)

    diffY = np.abs(lat - (attributeClone["yUL"] - 0.5 * cellsizeInput))
    yIdxSta = np.argmin(diffY)
    yIdxEnd = math.ceil(yIdxSta + attributeClone["rows"] / factor)

    return (
        factor,
        slice(max(xIdxSta - padding, 0), xIdxEnd + padding),
        slice(max(yIdxSta - padding, 0), yIdxEnd + padding),
    )


def interpolateToClone(cropData, lat, lon):
    lonClone = pcr.pcr2numpy(pcr.xcoordinate(pcr.boolean(1.0)), np.nan)[0, :]
    latClone = np.sort(pcr.pcr2numpy(pcr.ycoordinate(pcr.boolean(1.0)), np.nan)[:, 0])

    grid = xr.DataArray(
        cropData,
        dims=["latitude", "longitude"],
        coords=dict(latitude=lat, longitude=lon),
    ).sortby("latitude")
    grid = pyinterp.backends.xarray.Grid2D(grid, geodetic=False)

    mx, my = np.meshgrid(lonClone, latClone, indexing="ij")
    interpolated = grid.bivariate(
        coords=dict(longitude=mx.ravel(), latitude=my.ravel()), num_threads=1
    )
    # numpy2pcr reads the raw buffer, so a flipped or transposed view would scramble the map
    return interpolated.reshape(mx.shape).T[::-1].copy(order="C")


def getNetcdfGridAttributes(f):
    lat = f.variables["lat"]
    lon = f.variables["lon"]
    cellsize = float(lat[0] - lat[1])
    return {
        "cellsize": cellsize,
        "rows": len(lat),
        "cols": len(lon),
        "xUL": float(lon[0]) - 0.5 * cellsize,
        "yUL": float(lat[0]) + 0.5 * cellsize,
    }


def getPCRasterGridAttributes(mapFile):
    # CSF raster header; PCRaster always writes it little-endian
    with open(mapFile, "rb") as f:
        header = f.read(132)
    if not header.startswith(b"RUU CROSS SYSTEM MAP FORMAT"):
        raise RuntimeError(f"{mapFile} is not a PCRaster map")
    xUL, yUL, rows, cols, cellsize = struct.unpack_from("<ddIId", header, 84)
    return {
        "cellsize": cellsize,
        "rows": rows,
        "cols": cols,
        "xUL": xUL,
        "yUL": yUL,
    }


def gdalwarpPCR(input, isInteger=False):
    # nearest-neighbour resample of a PCRaster map onto the clone grid
    cloneAtt = utils.getCloneAttributes()
    cellsize = cloneAtt["cellsize"]
    xmin = cloneAtt["xUL"]
    ymax = cloneAtt["yUL"]
    xmax = xmin + cloneAtt["cols"] * cellsize
    ymin = ymax - cloneAtt["rows"] * cellsize
    if isInteger:
        outputType, nodata, valueScale = gdal.GDT_Int32, -(2**31), pcr.Nominal
    else:
        outputType, nodata, valueScale = (
            gdal.GDT_Float64,
            -3.4028234663852886e38,
            pcr.Scalar,
        )
    warped = gdal.Warp(
        "",
        str(input),
        format="MEM",
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=cellsize,
        yRes=cellsize,
        outputType=outputType,
        dstNodata=nodata,
    )
    return pcr.numpy2pcr(valueScale, warped.ReadAsArray(), nodata)


def set_clone(cloneMapFileName=None):
    """Set the PCRaster clone for the whole process and return pcr.clone().

    - The first call with a clone map sets the clone.
    - Later calls with a map on the same grid change nothing (e.g. spin-up runs that
      rebuild the model).
    - Without a clone map the clone already set is returned; it raises if none is set.
    """

    def grid():
        # pcr.clone() follows later setclone calls, so its values are copied out
        c = pcr.clone()
        return c.nrRows(), c.nrCols(), c.cellSize(), c.west(), c.north()

    current_grid = grid()
    # pcr.clone() does not raise when no clone is set; it reports zero rows
    if current_grid[0] == 0:
        if cloneMapFileName is None:
            raise RuntimeError("no clone has been set to inherit; give a clone map")
        pcr.setclone(cloneMapFileName)
        logger.info(f"clone set from {cloneMapFileName}")
    elif cloneMapFileName is not None:
        pcr.setclone(cloneMapFileName)
        if grid() != current_grid:
            raise RuntimeError(
                f"the clone is already set to a different grid than {cloneMapFileName}"
            )
    return pcr.clone()


def fillWithWindowAverage(map, windowLengths, mask):
    # window lengths are in map units (degrees on a geographic clone)
    # filling stops once every mask cell is defined: values inside the mask
    # match the unmasked result, cells outside it may stay MV
    for windowLength in windowLengths:
        defined = pcr.scalar(pcr.defined(map))
        if pcr.cellvalue(pcr.mapmaximum(defined), 1)[0] == 0:
            break
        defined = pcr.ifthen(mask, defined)
        minDefined, valid = pcr.cellvalue(pcr.mapminimum(defined), 1)
        if not valid or minDefined == 1:
            break
        map = pcr.cover(map, pcr.windowaverage(map, windowLength))
    return map


def regridData2FinerGrid(rescaleFac, coarse, MV):
    if rescaleFac == 1:
        return coarse
    nr, nc = np.shape(coarse)

    fine = (
        np.zeros(nr * nc * rescaleFac * rescaleFac).reshape(
            nr * rescaleFac, nc * rescaleFac
        )
        + MV
    )

    ii = -1
    nrF, ncF = np.shape(fine)
    for i in range(0, nrF):
        if i % rescaleFac == 0:
            ii += 1
        fine[i, :] = coarse[ii, :].repeat(rescaleFac)

    return fine


def findLastYearInNCTime(ncTimeVariable):

    last_datetime = nc.num2date(
        ncTimeVariable[len(ncTimeVariable) - 1],
        ncTimeVariable.units,
        ncTimeVariable.calendar,
    )

    return last_datetime.year


def findFirstYearInNCTime(ncTimeVariable):

    first_datetime = nc.num2date(
        ncTimeVariable[0], ncTimeVariable.units, ncTimeVariable.calendar
    )

    return first_datetime.year
