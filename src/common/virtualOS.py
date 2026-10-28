import calendar
import datetime
import functools
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import netCDF4 as nc
import numpy as np
import pcraster as pcr
import pyinterp.backends.xarray
import xarray as xr
import zarr

logger = logging.getLogger(__name__)

# file cache to reduce opening and closing files
filecache = dict()
# NcGrid per file in filecache; an entry must be dropped when its file is closed
_gridcache = dict()

# global variables
MV = 1e20
smallNumber = 1e-39

pi = math.pi

# netCDF file suffixes that can be used
netcdf_suffixes = (".nc4", ".nc")


def _retryIfRemote(read, maxTries=5):
    # only remote (OPeNDAP) sources are retried; local files are read once
    @functools.wraps(read)
    def wrapper(ncFile, *args, **kwargs):
        if not str(ncFile).startswith("http"):
            return read(ncFile, *args, **kwargs)
        for attempt in range(1, maxTries + 1):
            try:
                return read(ncFile, *args, **kwargs)
            except (OSError, RuntimeError) as error:
                if attempt == maxTries:
                    raise
                logger.warning(f"Re-try {attempt}/{maxTries} to read {ncFile}: {error}")

    return wrapper


def _resolveVarName(f, varName):
    # "automatic" picks the last variable that is neither a dimension nor a lat/lon axis
    varName = str(varName)
    if varName == "automatic":
        axes = set(f.dimensions) | {"lat", "lon", "latitude", "longitude"}
        return [var for var in f.variables if var not in axes][-1]
    # PCR-GLOBWB variable names and the names some input files use instead
    aliases = {
        "evapotranspiration": "referencePotET",
        "kc": "Cropcoefficient",
        "interceptCapInput": "Interceptioncapacity",
        "coverFractionInput": "Coverfraction",
        "fracVegCover": "vegetation_fraction",
        "minSoilDepthFrac": "minRootDepthFraction",
        "maxSoilDepthFrac": "maxRootDepthFraction",
        "arnoBeta": "arnoSchemeBeta",
    }
    if varName not in f.variables and aliases.get(varName) in f.variables:
        return aliases[varName]
    return varName


@dataclass(frozen=True, eq=False)
class NcGrid:
    # axes as read from the file, unconverted: a dtype change can move the argmin
    # in cloneWindow and so change which cells are read
    lat: np.ndarray
    lon: np.ndarray

    @classmethod
    def fromDataset(cls, f):
        # latitude/longitude take precedence over lat/lon when a file has both
        v = f.variables
        lat = v["latitude"] if "latitude" in v else v["lat"]
        lon = v["longitude"] if "longitude" in v else v["lon"]
        return cls(lat[:], lon[:])

    @functools.cached_property
    def attributes(self):
        cellsize = float(self.lat[0] - self.lat[1])
        return {
            "cellsize": cellsize,
            "rows": len(self.lat),
            "cols": len(self.lon),
            "xUL": float(self.lon[0]) - 0.5 * cellsize,
            "yUL": float(self.lat[0]) + 0.5 * cellsize,
        }

    def cloneWindow(self):
        """Return (yslice, xslice, factor) covering the clone; factor is the number of
        clone cells per file cell, for regridData2FinerGrid."""
        if isSameClone(self.attributes):
            return slice(None), slice(None), 1

        clone = getCloneAttributes()
        cellsize = self.attributes["cellsize"]
        factor = int(round(cellsize / float(clone["cellsize"])))
        if factor > 1:
            logger.debug(f"Resampling cell size {cellsize} to {clone['cellsize']}")

        xIdxSta = np.argmin(np.abs(self.lon - (clone["xUL"] + 0.5 * cellsize)))
        yIdxSta = np.argmin(np.abs(self.lat - (clone["yUL"] - 0.5 * cellsize)))
        return (
            slice(yIdxSta, math.ceil(yIdxSta + clone["rows"] / factor)),
            slice(xIdxSta, math.ceil(xIdxSta + clone["cols"] / factor)),
            factor,
        )


def _ncGrid(ncFile):
    if ncFile not in _gridcache:
        if ncFile not in filecache:
            filecache[ncFile] = nc.Dataset(ncFile)
        _gridcache[ncFile] = NcGrid.fromDataset(filecache[ncFile])
    return _gridcache[ncFile]


def _timeIndex(f, dateInput, useDoy):
    # returns (idx, date); date is the one actually looked up, after year clamping,
    # and is what daily_per_monthly_file uses to decide when to close the file
    time = f.variables["time"]
    ncFile = f.filepath()
    date = dateInput

    if dateInput is None:
        logger.debug("Using the first time step in the netcdf file.")
        idx = 0
        if len(time) > 1:
            logger.warning(
                "NOTE that there are more than one time steps in the netcdf file."
            )
    elif useDoy == "Yes":
        logger.debug("Finding the date from climatology doy index (1 to 366)")
        idx = int(dateInput) - 1
    # needed for netCDF files that contain only 12 monthly values (e.g. cropCoefficientWaterNC)
    elif useDoy == "month":
        logger.debug("Finding the date from climatology month index (1 to 12)")
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
        idx = int(date.month) - 1
    else:
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
        date = datetime.datetime(date.year, date.month, date.day)
        if useDoy == "yearly":
            date = datetime.datetime(date.year, 1, 1)
        if useDoy == "monthly":
            date = datetime.datetime(date.year, date.month, 1)
        if useDoy in (
            "yearly",
            "monthly",
            "daily_seasonal",
            "daily",
            "daily_per_monthly_file",
        ):
            # if the desired year is not available, use the first or last available year
            firstYear = findFirstYearInNCTime(time)
            lastYear = findLastYearInNCTime(time)
            year = min(max(date.year, firstYear), lastYear)
            if year != date.year:
                day = date.day
                if (
                    date.month == 2
                    and day == 29
                    and calendar.isleap(date.year)
                    and not calendar.isleap(year)
                ):
                    day = 28
                date = datetime.datetime(year, date.month, day)
                logger.warning(
                    f"Date {dateInput} NOT available in {ncFile}; "
                    f"using {date.year}-{date.month}-{date.day} instead."
                )
        try:
            idx = nc.date2index(date, time, calendar=time.calendar, select="exact")
            logger.debug(f"{date:%Y-%m-%d} found in netcdf; using select='exact'")
        except Exception:
            logger.debug(f"{date:%Y-%m-%d} not found; select='exact' not possible")
            if useDoy == "daily":
                select = "after"
                idx = nc.date2index(date, time, calendar=time.calendar, select=select)
            else:
                try:
                    select = "before"
                    idx = nc.date2index(
                        date, time, calendar=time.calendar, select=select
                    )
                except Exception:
                    select = "after"
                    idx = nc.date2index(
                        date, time, calendar=time.calendar, select=select
                    )
            logger.warning(
                f"Date {date.year}-{date.month}-{date.day} 00:00:00 NOT available in {ncFile}; "
                f"the '{select}' option is used, giving "
                f"{nc.num2date(time[int(idx)], time.units, time.calendar)}."
            )

    idx = int(idx)
    logger.debug(f"Using the date index {idx}")
    logger.debug(
        f"Using the datetime {nc.num2date(time[idx], time.units, time.calendar)}"
    )
    return idx, date


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


def readDownscalingZarr(
    ncFile,
    dateInput=None,
    useDoy=None,
    LatitudeLongitude=True,
    specificFillValue=None,
):
    f = zarr.convenience.open(ncFile)

    keys = sorted(f.keys())

    yRef = "lat"
    xRef = "lon"
    if LatitudeLongitude:
        if "latitude" in keys:
            yRef = "latitude"
        if "longitude" in keys:
            xRef = "longitude"

    dims = ["time", xRef, yRef]
    varName = [item for item in keys if item not in dims][0]

    attributeClone = getCloneAttributes()
    cellsizeClone = attributeClone["cellsize"]
    rowsClone = attributeClone["rows"]
    colsClone = attributeClone["cols"]
    xULClone = attributeClone["xUL"]
    yULClone = attributeClone["yUL"]
    # attributes of the input (netCDF)
    cellsizeInput = f[yRef][0] - f[yRef][1]
    cellsizeInput = float(cellsizeInput)

    factor = int(round(float(cellsizeInput) / float(cellsizeClone)))
    diffX = np.abs(f[xRef][:] - (xULClone + 0.5 * cellsizeInput))
    xIdxSta = np.argmin(diffX)
    xIdxEnd = math.ceil(xIdxSta + colsClone / factor)
    xslice = slice(xIdxSta, xIdxEnd)

    diffY = np.abs(f[yRef][:] - (yULClone - 0.5 * cellsizeInput))
    yIdxSta = np.argmin(diffY)
    yIdxEnd = math.ceil(yIdxSta + rowsClone / factor)
    yslice = slice(yIdxSta, yIdxEnd)

    timeID = dateInput - 1

    cropData = f[varName].get_basic_selection((timeID, xslice, yslice))[:]
    cropData = np.nan_to_num(cropData).T

    lon = pcr.pcr2numpy(pcr.xcoordinate(pcr.boolean(1.0)), np.nan)[0, :]
    lat = pcr.pcr2numpy(pcr.ycoordinate(pcr.boolean(1.0)), np.nan)[:, 0]
    cropData = xr.DataArray(
        cropData,
        dims=["latitude", "longitude"],
        coords=dict(longitude=lon, latitude=lat),
    ).sortby("latitude", ascending=False)

    cropData = cropData.values
    # convert to a PCRaster object
    if specificFillValue is not None:
        outPCR = pcr.numpy2pcr(pcr.Scalar, cropData, float(specificFillValue))
    else:
        try:
            outPCR = pcr.numpy2pcr(pcr.Scalar, cropData, float(f[varName].fill_value))
        except Exception:
            outPCR = pcr.numpy2pcr(pcr.Scalar, cropData, float(MV))

    return outPCR


def readDownscalingMeteo(
    ncFile,
    varName="automatic",
    dateInput=None,
    useDoy=None,
    LatitudeLongitude=True,
    specificFillValue=None,
):
    # EHS (19 Apr 2013): convert a netCDF (tss) file to a PCRaster map with clone checking;
    # only works for square cells and if cellsizeClone <= cellsizeInput

    if ncFile not in filecache:
        filecache[ncFile] = nc.Dataset(ncFile)
    f = filecache[ncFile]
    varName = _resolveVarName(f, varName)
    logger.debug(f"reading variable: {varName} from the file: {ncFile}")
    grid = _ncGrid(ncFile)
    ncLat, ncLon = grid.lat, grid.lon

    idx, _ = _timeIndex(f, dateInput, useDoy)

    attributeClone = getCloneAttributes()
    cellsizeClone = attributeClone["cellsize"]
    rowsClone = attributeClone["rows"]
    colsClone = attributeClone["cols"]
    xULClone = attributeClone["xUL"]
    yULClone = attributeClone["yUL"]
    # attributes of the input (netCDF)
    cellsizeInput = grid.attributes["cellsize"]

    # correction for files with an additional level dimension (e.g. WFDEI forcing: time, height/level, lat, lon)
    if f.variables[varName].ndim == 4:
        logger.warning("%s (%s) is 4D; last two dims read as lat/lon", ncFile, varName)
        # file with an additional layer/dimension (original data)
        cropData = f.variables[varName][int(idx), 0, :, :]
    else:
        # standard netCDF file (original data)
        cropData = f.variables[varName][int(idx), :, :]

    # needed in regridData2FinerGrid
    factor = 1
    factor = int(round(float(cellsizeInput) / float(cellsizeClone)))

    # crop to the clone map
    minX = min(abs(ncLon[:] - (xULClone + 0.5 * cellsizeInput)))
    xIdxSta = (
        int(np.where(abs(ncLon[:] - (xULClone + 0.5 * cellsizeInput)) == minX)[0]) - 1
    )
    if xIdxSta == -1:
        xIdxSta = 0

    xIdxEnd = int(math.ceil((xIdxSta + 1) + colsClone / (factor))) + 1

    minY = min(abs(ncLat[:] - (yULClone - 0.5 * cellsizeInput)))

    yIdxSta = (
        int(np.where(abs(ncLat[:] - (yULClone - 0.5 * cellsizeInput)) == minY)[0]) - 1
    )

    yIdxEnd = int(math.ceil((yIdxSta + 1) + rowsClone / (factor))) + 1

    # retrieve the data slice from the netCDF file

    if f.variables[varName].ndim == 4:
        logger.warning("%s (%s) is 4D; last two dims read as lat/lon", ncFile, varName)
        # file with an additional layer
        cropData = f.variables[varName][int(idx), 0, yIdxSta:yIdxEnd, xIdxSta:xIdxEnd]
    else:
        # standard netCDF file
        cropData = f.variables[varName][int(idx), yIdxSta:yIdxEnd, xIdxSta:xIdxEnd]
    lon = pcr.pcr2numpy(pcr.xcoordinate(pcr.boolean(1.0)), np.nan)[0, :]
    lat = np.sort(pcr.pcr2numpy(pcr.ycoordinate(pcr.boolean(1.0)), np.nan)[:, 0])

    array = xr.DataArray(
        cropData,
        dims=["latitude", "longitude"],
        coords=dict(
            latitude=ncLat[yIdxSta:yIdxEnd],
            longitude=ncLon[xIdxSta:xIdxEnd],
        ),
    ).sortby("latitude")

    array = pyinterp.backends.xarray.Grid2D(array, geodetic=False)
    mx, my = np.meshgrid(lon, lat, indexing="ij")
    cropData = array.bivariate(
        coords=dict(longitude=mx.ravel(), latitude=my.ravel()), num_threads=1
    )
    cropData = cropData.reshape(mx.shape).T

    cropData = xr.DataArray(
        cropData,
        dims=["latitude", "longitude"],
        coords=dict(longitude=lon, latitude=lat),
    ).sortby("latitude", ascending=False)

    cropData = cropData.values
    if specificFillValue is not None:
        outPCR = pcr.numpy2pcr(pcr.Scalar, cropData, float(specificFillValue))
    else:
        try:
            outPCR = pcr.numpy2pcr(
                pcr.Scalar, cropData, float(f.variables[varName]._FillValue)
            )
        except Exception:
            outPCR = pcr.numpy2pcr(pcr.Scalar, cropData, float(MV))
    return outPCR


def checkVariableInNC(ncFile, varName):

    logger.debug(f"Checking whether {varName} is defined in {ncFile}")

    if ncFile not in filecache:
        filecache[ncFile] = nc.Dataset(ncFile)
    return str(varName) in filecache[ncFile].variables


@_retryIfRemote
def netcdf2PCRobjCloneWithoutTime(
    ncFile,
    varName,
    LatitudeLongitude=True,
    specificFillValue=None,
    absolutePath=None,
):

    if absolutePath is not None:
        ncFile = getFullPath(ncFile, absolutePath)

    # EHS (19 Apr 2013): convert a netCDF (tss) file to a PCRaster map with clone checking;
    # only works for square cells and if cellsizeClone <= cellsizeInput

    if ncFile not in filecache:
        filecache[ncFile] = nc.Dataset(ncFile)
    f = filecache[ncFile]
    varName = _resolveVarName(f, varName)
    logger.debug(f"reading variable: {varName} from the file: {ncFile}")

    yslice, xslice, factor = _ncGrid(ncFile).cloneWindow()

    cropData = f.variables[varName][yslice, xslice]

    # convert to a PCRaster object
    if specificFillValue is not None:
        outPCR = pcr.numpy2pcr(
            pcr.Scalar,
            regridData2FinerGrid(factor, cropData, float(specificFillValue)),
            float(specificFillValue),
        )
    else:
        try:
            outPCR = pcr.numpy2pcr(
                pcr.Scalar,
                regridData2FinerGrid(
                    factor, cropData, float(f.variables[varName]._FillValue)
                ),
                float(f.variables[varName]._FillValue),
            )
        except Exception:
            outPCR = pcr.numpy2pcr(
                pcr.Scalar,
                regridData2FinerGrid(
                    factor, cropData, float(f.variables[varName].missing_value)
                ),
                float(f.variables[varName].missing_value),
            )

    return outPCR


@_retryIfRemote
def netcdf2PCRobjClone(
    ncFile,
    varName="automatic",
    dateInput=None,
    useDoy=None,
    LatitudeLongitude=True,
    specificFillValue=None,
):
    # EHS (19 Apr 2013): convert a netCDF (tss) file to a PCRaster map with clone checking;
    # only works for square cells and if cellsizeClone <= cellsizeInput

    if ncFile not in filecache:
        filecache[ncFile] = nc.Dataset(ncFile)
    f = filecache[ncFile]
    varName = _resolveVarName(f, varName)
    logger.debug(f"reading variable: {varName} from the file: {ncFile}")

    idx, date = _timeIndex(f, dateInput, useDoy)

    yslice, xslice, factor = _ncGrid(ncFile).cloneWindow()

    # retrieve the data slice from the netCDF file
    if f.variables[varName].ndim == 4:
        logger.warning("%s (%s) is 4D; last two dims read as lat/lon", ncFile, varName)
        # file with an additional layer
        cropData = f.variables[varName][int(idx), 0, yslice, xslice]
    else:
        # standard netCDF file
        cropData = f.variables[varName][int(idx), yslice, xslice]

    # convert to a PCRaster object
    if specificFillValue is not None:
        outPCR = pcr.numpy2pcr(
            pcr.Scalar,
            regridData2FinerGrid(factor, cropData, float(specificFillValue)),
            float(specificFillValue),
        )
    else:
        try:
            outPCR = pcr.numpy2pcr(
                pcr.Scalar,
                regridData2FinerGrid(
                    factor, cropData, float(f.variables[varName]._FillValue)
                ),
                float(f.variables[varName]._FillValue),
            )
        except Exception:
            outPCR = pcr.numpy2pcr(
                pcr.Scalar,
                regridData2FinerGrid(
                    factor, cropData, float(f.variables[varName].missing_value)
                ),
                float(f.variables[varName].missing_value),
            )

    if useDoy == "daily_per_monthly_file":
        # close the file on the last day of the month and remove it from the cache
        tomorrow = date + datetime.timedelta(days=1)
        if tomorrow.day == 1:
            f.close()
            del filecache[ncFile]
            _gridcache.pop(ncFile, None)

    return outPCR


def writePCRmapToDir(v, outFileName, outDir):
    # v: input map file name or value; if the input map and the clone have different
    # grids, the map is resampled
    fullFileName = getFullPath(outFileName, outDir)
    logger.debug(f"Writing a pcraster map to : {fullFileName}")
    pcr.report(v, fullFileName)


def readPCRmapClone(
    v,
    tmpDir,
    absolutePath=None,
    isLddMap=False,
    cover=None,
    isNomMap=False,
):
    # v: input map file name or value; if the input map and the clone have different
    # grids, the map is resampled
    logger.debug(f"read file/value: {v}")

    if v == "None":

        PCRmap = None

    elif not re.match(r"[0-9.-]*$", v):
        if absolutePath is not None:
            v = getFullPath(v, absolutePath)

        this_is_a_netcdf_file = False
        if v.endswith(".nc") or v.endswith(".nc4"):
            this_is_a_netcdf_file = True

        if this_is_a_netcdf_file:

            logger.debug(f"read netcdf file: {v}")

            try:
                # netCDF file without time
                PCRmap = netcdf2PCRobjCloneWithoutTime(ncFile=v, varName="automatic")
            except Exception:
                # netCDF file with time
                PCRmap = netcdf2PCRobjClone(
                    ncFile=v,
                    varName="automatic",
                    dateInput=None,
                    useDoy=None,
                )

        else:

            # PCRaster format

            sameClone = isSameClone(getPCRasterGridAttributes(v))
            if sameClone:
                PCRmap = pcr.readmap(v)
            else:
                # resample using GDAL
                output = f"{tmpDir}temp.map"
                _ = gdalwarpPCR(v, output, tmpDir, isLddMap, isNomMap)
                # read and delete the temporary file
                PCRmap = pcr.readmap(output)
                if os.path.isdir(tmpDir):
                    shutil.rmtree(tmpDir)
                os.makedirs(tmpDir)
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


def isSameClone(attributeInput):
    # rounded to 6 significant digits: mapattr prints only 6, and it absorbs floating point drift in derived corners
    attributeClone = getCloneAttributes(arcDegree=False)
    return all(
        float(f"{attributeClone[key]:.6g}") == float(f"{attributeInput[key]:.6g}")
        for key in attributeClone
    )


def getPCRasterGridAttributes(mapFile):
    result = subprocess.run(
        ["mapattr", "-p", str(mapFile)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot read the map attributes of {mapFile} with mapattr: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    attributes = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            attributes[parts[0]] = parts[-1]
    return {
        "cellsize": float(attributes["cell_length"]),
        "rows": int(attributes["rows"]),
        "cols": int(attributes["columns"]),
        "xUL": float(attributes["xUL"]),
        "yUL": float(attributes["yUL"]),
    }


def gdalwarpPCR(input, output, tmpDir, isLddMap=False, isNominalMap=False):
    # 19 Mar 2013, Edwin H. Sutanudjaja; all input maps must be PCRaster maps
    # remove temporary files
    co = f"rm {tmpDir}*.*"
    cOut, err = subprocess.Popen(
        co, stdout=subprocess.PIPE, stderr=open(os.devnull), shell=True
    ).communicate()
    # convert files to tif
    co = f"gdal_translate -ot Float64 {input} {tmpDir}tmp_inp.tif"
    if isLddMap:
        co = f"gdal_translate -ot Int32 {input} {tmpDir}tmp_inp.tif"
    if isNominalMap:
        co = f"gdal_translate -ot Int32 {input} {tmpDir}tmp_inp.tif"
    cOut, err = subprocess.Popen(
        co, stdout=subprocess.PIPE, stderr=open(os.devnull), shell=True
    ).communicate()
    cloneAtt = getCloneAttributes()
    xmin = cloneAtt["xUL"]
    ymin = cloneAtt["yUL"] - cloneAtt["rows"] * cloneAtt["cellsize"]
    xmax = cloneAtt["xUL"] + cloneAtt["cols"] * cloneAtt["cellsize"]
    ymax = cloneAtt["yUL"]
    xres = cloneAtt["cellsize"]
    yres = cloneAtt["cellsize"]
    te = f"-te {xmin} {ymin} {xmax} {ymax} "
    tr = f"-tr {xres} {yres} "
    co = (
        "gdalwarp "
        + te
        + tr
        + " -srcnodata -3.4028234663852886e+38 -dstnodata mv "
        + str(tmpDir)
        + "tmp_inp.tif "
        + str(tmpDir)
        + "tmp_out.tif"
    )
    cOut, err = subprocess.Popen(
        co, stdout=subprocess.PIPE, stderr=open(os.devnull), shell=True
    ).communicate()
    co = f"gdal_translate -of PCRaster {tmpDir}tmp_out.tif {output}"
    cOut, err = subprocess.Popen(
        co, stdout=subprocess.PIPE, stderr=open(os.devnull), shell=True
    ).communicate()

    co = f"rm {tmpDir}tmp*.*"
    cOut, err = subprocess.Popen(
        co, stdout=subprocess.PIPE, stderr=open(os.devnull), shell=True
    ).communicate()


def getFullPath(inputPath, absolutePath, completeFileName=True):
    # 19 Mar 2013, Edwin H. Sutanudjaja: get the full absolute path of a folder or file

    inputPath = str(inputPath).replace("\\", "/")
    absolutePath = str(absolutePath).replace("\\", "/")

    # suffixes that can be used
    suffix = (
        "/",
        "_",
        ".nc4",
        ".map",
        ".nc",
        ".dat",
        ".txt",
        ".asc",
        ".ldd",
        ".tbl",
        ".001",
        ".002",
        ".003",
        ".004",
        ".005",
        ".006",
        ".007",
        ".008",
        ".009",
        ".010",
        ".011",
        ".012",
    )

    if (
        inputPath.startswith("/")
        or str(inputPath)[1] == ":"
        or inputPath.startswith("http")
    ):
        fullPath = str(inputPath)
    else:
        if absolutePath.endswith("/"):
            absolutePath = str(absolutePath)
        else:
            absolutePath = f"{absolutePath}/"
        fullPath = str(absolutePath) + str(inputPath)

    if completeFileName:
        if fullPath.endswith(suffix):
            fullPath = str(fullPath)
        else:
            fullPath = f"{fullPath}/"

    return fullPath


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


def getCloneAttributes(arcDegree=True):
    clone = pcr.clone()
    cellsize = clone.cellSize()
    if arcDegree:
        cellsize = round(cellsize * 360000.0) / 360000.0
    return {
        "cellsize": cellsize,
        "rows": clone.nrRows(),
        "cols": clone.nrCols(),
        "xUL": clone.west(),
        "yUL": clone.north(),
    }


def getMapTotal(mapFile):
    """outputs the sum of all values in a map file"""

    total, valid = pcr.cellvalue(pcr.maptotal(mapFile), 1)
    return total


def getMinMaxMean(mapFile, ignoreEmptyMap=False):
    mn = pcr.cellvalue(pcr.mapminimum(mapFile), 1)[0]
    mx = pcr.cellvalue(pcr.mapmaximum(mapFile), 1)[0]
    nrValues = pcr.cellvalue(pcr.maptotal(pcr.scalar(pcr.defined(mapFile))), 1)[0]
    if nrValues == 0.0 and ignoreEmptyMap:
        logger.warning("map is empty")
        return 0.0, 0.0, 0.0
    elif nrValues == 0.0 and not ignoreEmptyMap:
        logger.warning("map is empty")
        return 0.0, 0.0, 0.0
    else:
        return mn, mx, (getMapTotal(mapFile) / nrValues)


def getMapVolume(mapFile, cellareaFile):
    """returns the sum of all grid cell values"""
    volume = mapFile * cellareaFile
    return getMapTotal(volume) / 1


def secondsPerDay():
    return float(3600 * 24)


def getValDivZero(x, y, y_lim=smallNumber, z_def=0.0):
    # division that may involve a zero denominator, in which case a default value is used:
    # x/y = z if y > y_lim, and z_def (zero by default) if y <= y_lim, where y_lim -> 0
    return pcr.ifthenelse(y > y_lim, x / pcr.max(y_lim, y), z_def)


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


def waterBalanceCheck(
    fluxesIn,
    fluxesOut,
    preStorages,
    endStorages,
    processName,
    PrintOnlyErrors,
    dateStr,
    threshold=1e-5,
    landmask=None,
):
    """Returns the water balance for a list of input, output, and storage map files"""

    inMap = pcr.spatial(pcr.scalar(0.0))
    outMap = pcr.spatial(pcr.scalar(0.0))
    dsMap = pcr.spatial(pcr.scalar(0.0))

    for fluxIn in fluxesIn:
        inMap += fluxIn
    for fluxOut in fluxesOut:
        outMap += fluxOut
    for preStorage in preStorages:
        dsMap += preStorage
    for endStorage in endStorages:
        dsMap -= endStorage

    a, b, c = getMinMaxMean(inMap + dsMap - outMap)
    if abs(a) > threshold or abs(b) > threshold:
        if PrintOnlyErrors:

            bar = "#" * 142
            logger.error(
                f"\n\n{bar}\nWARNING !!!!!!!! Water Balance Error {processName} "
                f"Min {a:f} Max {b:f} Mean {c:f}\n{bar}\n\n\n\n"
            )


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


def deg2rad(a):

    return a * pi / 180.0


def rad2deg(a):

    return a * 180.0 / pi
