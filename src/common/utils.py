import calendar
import datetime
import logging
import math
import os
import shutil
import sys
import types
from copy import deepcopy

import numpy as np
import pcraster as pcr

logger = logging.getLogger(__name__)

NoneType = type(None)

smallNumber = 1e-39
very_small_number = 1.0e-12

pi = math.pi


def isSameClone(attributeInput):
    # rounded to 6 significant digits: mapattr prints only 6, and it absorbs floating point drift in derived corners
    attributeClone = getCloneAttributes(arcDegree=False)
    return all(
        float(f"{attributeClone[key]:.6g}") == float(f"{attributeInput[key]:.6g}")
        for key in attributeClone
    )


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
    """
getValDivZero: divides PCRaster fields and returns a default value where the \
denominator is too small, to avoid dividing by zero:

    z = x / y           if y > y_lim
    z = z_def           if y <= y_lim

    Input:
    ======
    x:                  numerator;
    y:                  denominator;
    y_lim:              value against which the denominator y is tested;
                        default is smallNumber;
    z_def:              value returned where y <= y_lim; default is zero.

    Output:
    =======
    z:                  result of the fraction x / y.

"""
    return pcr.ifthenelse(y > y_lim, x / pcr.max(y_lim, y), z_def)


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


def deg2rad(a):

    return a * pi / 180.0


def rad2deg(a):

    return a * 180.0 / pi


def sum_list(list_in):
    """
sum_list: generic function that returns the sum of a list of which \
the entries can be summed (integers, floats, arrays etc.).

"""
    # sum all entries of a list-like object
    return sum(list_in)


def max_dicts(dict_in):
    """
max_dicts: generic function that returns the maximum pcr map from a collection \
of arrays contained in a dictionary.

"""
    map_maximum = None

    # maximum, comparing maps in pairs
    for key, values in dict_in.items():
        if isinstance(map_maximum, type(None)):
            map_maximum = values

        else:
            map_maximum = pcr.max(values, map_maximum)

    return map_maximum


def get_decision(question_str, possible_outcomes):
    """
    get_decision: function that gets user input interactively.

        Input:
        ======
        question_str:       string specifying the options to the user.
                            possible answers are taken from the possible outcomes;
        possible_outcomes:  possible outcomes organized as a dictionary with
                            the answers as keys and the related outcome.

        Output:
        =======
        outcome:            the resulting outcome.

    """

    options = list(possible_outcomes.keys())
    option_str = "("
    for option in options:
        option_str = str.join("", (option_str, option, ", "))
    option_str = option_str.rstrip(", ")
    option_str = str.join("", (option_str, ")\n> "))
    question_str = str.join(" ", (question_str.rstrip(), option_str))

    outcome = None
    selected_option = ""
    while selected_option not in options:
        selected_option = input(question_str)

    outcome = possible_outcomes[selected_option]

    return outcome


def convert_string_to_list(s, separators=[",", ";"]):
    """

convert_string_to_list: function that converts a string with separators into \
a (nested) list.

"""

    if not isinstance(s, str):

        sys.exit("function requires a string to split it in smaller parts")

    result_list = [s]
    sep_level = 0

    while len(separators) > 0:

        separator = separators.pop()

        # split each entry into sub-levels
        for ix in range(len(result_list)):
            ss = result_list[ix]
            result_list[ix] = ss.split(sep=separator)

        # avoid nesting the first list
        if sep_level == 0:
            result_list = result_list[0][:]

        sep_level = sep_level + 1

    return result_list


def pcr_get_statistics(pcrfield, np_stat_funcs=[np.average, np.min, np.max], mv=-999.9):
    """
pcr_get_statistics: returns the statistics of all values in a PCRaster field \
as specified by numpy functions in the np_stat_funcs list with the missing values \
masked out.

Returns a dictionary with the name of the numpy functions as keys and the count

"""
    a = pcr.pcr2numpy(pcr.scalar(pcrfield), mv).astype(np.float32)
    a = a[a != mv]

    if a.size > 0:
        stat_val = dict((str(f.__name__), f(a)) for f in np_stat_funcs)
    else:
        stat_val = dict((str(f.__name__), mv) for f in np_stat_funcs)
    stat_val["count"] = a.size

    return stat_val


def pcr_same_map(map_a: pcr.Field, map_b: pcr.Field) -> bool:
    """Returns True if both maps have the same values and missing values."""
    differs = pcr.ifthenelse(
        pcr.defined(map_a) & pcr.defined(map_b),
        map_a != map_b,
        pcr.defined(map_a) != pcr.defined(map_b),
    )
    return pcr.cellvalue(pcr.mapmaximum(pcr.scalar(differs)), 1)[0] == 0


def pcr_share_identical_maps(maps: dict[str, pcr.Field]) -> dict[str, pcr.Field]:
    """
    Returns the maps with identical maps (same values and missing values) replaced
    by one object: each map takes the first identical map before it, in the order of
    maps. Identical maps can then be recognized by object, e.g. to group sources
    that share a zone map.
    """

    # identical is transitive, so each map is only compared with the distinct maps
    distinct_maps = []
    shared_maps = {}
    for key, map_a in maps.items():
        for map_b in distinct_maps:
            if map_a is map_b or pcr_same_map(map_a, map_b):
                shared_maps[key] = map_b
                break
        else:
            distinct_maps.append(map_a)
            shared_maps[key] = map_a

    return shared_maps


# default file extensions
file_extensions = {
    ".map": "pcraster",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".tif": "pcraster",
    "": "",
}


def compose_filename(filename, path, *args):
    """
compose_filename: function that checks whether the filename is an absolute path, \
and if not merges it with the path provided, normalizes the path and tests if \
the file exists. Returns the resulting the filename.
"""

    # make the file name absolute if the file exists
    if os.path.isfile(filename):
        filename = os.path.abspath(filename)
    else:
        filename = os.path.join(path, filename)

    filename = os.path.normpath(filename)

    # substitute any additional arguments
    if args != ():

        if not isinstance(args, tuple):
            args = tuple(args)

        if "%" in filename:
            try:
                filename = filename % (args)
            except Exception:
                logger.warning(
                    "additional arguments could not be converted into the file name %s",
                    filename,
                )

    return filename, os.path.isfile(filename)


def file_is_nc(filename):
    """
    file_is_ncfile: tests if the file extension matches that of a netCDF file;
    returns True if this is the case.
    """

    file_ext = os.path.splitext(filename)[1]

    return file_extensions[file_ext] == "netcdf"


def file_is_pcr(filename):
    """
    file_is_ncfile: tests if the file extension matches that of a netCDF file;
    returns True if this is the case.
    """

    file_ext = os.path.splitext(filename)[1]

    return file_extensions[file_ext] == "pcraster"


def cloneAttributes():
    clone = pcr.clone()
    rows, cols, cellsize = clone.nrRows(), clone.nrCols(), clone.cellSize()
    return types.SimpleNamespace(
        numberRows=rows,
        numberCols=cols,
        xResolution=cellsize,
        yResolution=cellsize,
        xLL=clone.west(),
        xUR=clone.west() + cols * cellsize,
        yLL=clone.north() - rows * cellsize,
        yUR=clone.north(),
    )


def checkCoordinate(v1, v2, delta_v1, delta_v2):
    # check coordinates using delta_v, the minimum resolution
    delta_v = min(abs(delta_v1), abs(delta_v2))
    return abs(v1 - v2) < 0.5 * delta_v


def compareSpatialAttributes(
    sourceSpatialDataSet, targetSpatialDataSet, resamplePrecision=1.0e-5
):
    """Compares the attributes of two spatial datasets defined by the spatialAttributes instance\
 taking the second input as target.\
 """
    xResampleRatio = sourceSpatialDataSet.xResolution / targetSpatialDataSet.xResolution
    yResampleRatio = sourceSpatialDataSet.yResolution / targetSpatialDataSet.yResolution
    # check whether the corner coordinates of the target area lie within the source extent
    sameCoordinates = {}
    fitsExtent = True
    for coordKey in ["xLL", "xUR", "yLL", "yUR"]:
        resolutionKey = f"{coordKey[:1]}Resolution"
        sameCoordinates[coordKey] = checkCoordinate(
            getattr(sourceSpatialDataSet, coordKey),
            getattr(targetSpatialDataSet, coordKey),
            getattr(sourceSpatialDataSet, resolutionKey),
            getattr(targetSpatialDataSet, resolutionKey),
        )
        fitsExtent &= sameCoordinates[coordKey]
    # if the corners do not match, determine whether the map fits the extent
    sameResolution = (
        abs(1 - xResampleRatio) < resamplePrecision
        and abs(1 - yResampleRatio) < resamplePrecision
    )

    # output: whether the map must be clipped or warped, and whether it must be rescaled
    return fitsExtent, sameResolution, xResampleRatio, yResampleRatio


def get_weights_from_dates(dates):

    # weights of the time increments of dates over the total length; input is a
    # sequence of dates (list, numpy array), output is a dictionary with the dates
    # as keys

    seq_type = type(dates)

    d = deepcopy(dates)

    if seq_type == list:
        d = np.array(d)

    # flatten and sort the dates
    d = d.ravel()
    d.sort()

    w = np.zeros(d.size)

    # add a final date
    d = np.append(d, datetime.datetime(d[0].year + 1, d[0].month, d[0].day))

    # iterate over the weights to get the date differences
    delta_d = d[-1] - d[0]
    for ix in range(w.size):

        w[ix] = (d[ix + 1] - d[ix]) / delta_d

    # cast the weights to the desired type
    if seq_type == list:
        w = w.tolist()

    return dict((d[ix], w[ix]) for ix in range(len(w)))


def update_year_of_date(date, new_year, dates):
    """
update_year_of_date: function that finds the offset in days between the date \
specified and the same day in the new year specified and applies it subsequently \
to dates, which can be an iterable or a single date. Returns a copy of dates \
with the years changed accordingly.

"""
    seq_type = True
    if isinstance(dates, list):
        new_dates = dates[:]

    elif isinstance(dates, np.ndarray):
        new_dates = deepcopy(dates)
        new_dates = new_dates.ravel()

    else:
        new_dates = [dates]
        seq_type = False

    # check the data type
    if not isinstance(new_dates[0], datetime.datetime):
        "dates have not the correct datetime datetime format"

    date_offset = datetime.datetime(new_year, date.month, date.day) - datetime.datetime(
        date.year, date.month, date.day
    )

    for ix in range(len(new_dates)):

        new_dates[ix] = new_dates[ix] + date_offset

    # single entry or sequence
    if not seq_type:
        new_dates = new_dates[0]

    return new_dates


def substitute_years_of_dates(dates, date):
    """Returns a copy of a list of dates in which the years have been updated
    to include the year of the specified date"""

    # nearest date and available years
    date_index = get_date_index(date, dates, "nearest")
    nearest_date = dates[date_index]

    return update_year_of_date(nearest_date, date.year, dates)


def get_date_index(date, dates, date_selection_method, substituted_dates=False):
    """
    get_date_index: function that returns the position of the date specified in an array of dates,
    following the index date_selection_methodion method specified: exact, before, after, nearest.
    This function reproduces the date2index function of the netCDF4 package but
    circumvents the problem that the original function cannot read from a standard dictionary.

    """

    # array with time deltas and indices
    time_delta = dates.copy()

    if not isinstance(dates[0], type(date)):
        for ix in range(len(time_delta)):
            time_delta[ix] = datetime.datetime(
                time_delta[ix].year,
                time_delta[ix].month,
                time_delta[ix].day,
                time_delta[ix].hour,
                time_delta[ix].minute,
                time_delta[ix].second,
                time_delta[ix].microsecond,
            )
    time_delta = time_delta - date
    date_index = np.arange(time_delta.size)

    # masked array operation for Python 3.x and higher
    if isinstance(time_delta, np.ma.core.MaskedArray):
        time_delta = np.array(time_delta.tolist())

    # find the zero value
    if (
        np.size(dates[time_delta == datetime.timedelta(0)]) > 0
        and not substituted_dates
    ):
        return np.arange(dates.size)[time_delta == datetime.timedelta(0)][0]

    elif date_selection_method == "exact":
        return None

    elif date_selection_method == "before":
        mask = time_delta < datetime.timedelta(0)

        if time_delta[mask].size > 0:
            mask_value = time_delta[mask].max()
            return date_index[time_delta == mask_value][0]
        else:
            return None

    elif date_selection_method == "after":
        mask = time_delta > datetime.timedelta(0)
        if time_delta[mask].size > 0:
            mask_value = time_delta[mask].min()
            return date_index[time_delta == mask_value][0]
        else:
            return None

    elif date_selection_method == "nearest":
        time_delta = np.abs(time_delta)
        return np.arange(dates.size)[time_delta == time_delta.min()][0]

    else:
        sys.exit(
            "index date_selection_method %s is not allowed or does not yield a result"
            % date_selection_method
        )


def match_date_by_julian_number(date, dates, within_same_month=True):
    """
    Output:
    =======
match_date_by_julian_number: function that allows for date substitution in the water management \
module.

    Input:
    ======
    date:                   date provided;
    dates:                  array of available dates to which the provided
                            dates should be matched;
    date_selection_method:  date selection method, default value is 'exact';
                            can be 'exact', 'before', or 'after'.

    Output:
    =======
    date_index:             date index matching the sought date to the av-
                            ailable dates;
    matched_date:           the date in the available dates that matches the
                            date index;
    message_str:            message string on the date selction for subsequent
                            logging.

"""

    message_str = ""

    julian_day = get_julian_day_number(date)
    month = date.month
    year = date.year

    if isinstance(dates, list):
        dates = np.array(dates)

    julian_days = np.zeros(dates.shape, dtype=int)
    months = np.zeros(dates.shape, dtype=int)
    years = np.zeros(dates.shape, dtype=int)
    ix = 0
    for sdate in dates:
        julian_days[ix] = get_julian_day_number(sdate)
        months[ix] = sdate.month
        years[ix] = sdate.month
        ix = ix + 1

    ixs = np.arange(dates.size, dtype=int)
    ixs.shape = dates.shape

    devs = np.abs(julian_days - julian_day)

    if within_same_month and np.any(months == month):
        mask = months == month
    else:
        mask = months != 13
        if within_same_month:
            message_str = (
                f"; Warning: month {month} is not present in the provided dates"
            )

    if year in years:
        mask = mask & (years == year)

    # halt if the mask selects nothing
    if np.size(devs[mask]) > 0:

        devs = devs[mask]
        ixs = ixs[mask]

        min_dev = devs.min()
        date_index = int(ixs[devs == min_dev][0])

    matched_date = dates[date_index]
    message_str = str.join(
        "",
        (
            f"date {date} is matched with {matched_date} in the available dates",
            message_str,
        ),
    )

    return date_index, matched_date, message_str


def is_last_day_month(date):
    """returns if it is thelast of of the month"""
    today = date.day
    last_day = calendar.monthrange(date.year, date.month)[1]
    return today == last_day


def get_julian_day_number(date):
    """

    get_julian_day_number: returns the julian day number in the year.

        Input:
        ======
        date:               date as datetime date, datetime or cftime datetime.

        Output:
        =======
        julian_day:         integer number of days since 1 January of the
                            present year (zero-based).

    """
    # cftime dates from nc.num2date only subtract dates of their own calendar type
    return int((date - type(date)(date.year, 1, 1)).days)


def get_key(str_list):

    if isinstance(str_list, list):
        key = str_list[0]
    else:
        key = str(str_list)

    for ix in range(1, len(str_list)):

        if str_list[ix - 1] == "":
            d_str = ""
        else:
            d_str = "_"

        key = str.join(d_str, (key, str_list[ix]))

    return key


def get_zonal_total(local_values: pcr.Field, zones: pcr.Field | None) -> pcr.Field:
    """
get_zonal_total: function that computes the total of the local values per zone \
and assigns it to every cell in the zone.

    Input:
    ======
    local values:            local cell values as a scalar PCRaster field;
    zones:                   zones over which the totals are computed as
                             a nominal PCRaster field, or None if every cell
                             is its own zone.

    Output:
    =======
    totals:                  totals over the zones per cell as a scalar
                             PCRaster field.

"""
    # without zones, every cell is its own zone
    if zones is None:
        return local_values
    return pcr.areatotal(local_values, zones)


def get_zonal_minimum(local_values: pcr.Field, zones: pcr.Field | None) -> pcr.Field:
    """
get_zonal_minimum: function that computes the minimum of the local values per zone \
and assigns it to every cell in the zone.

    Input:
    ======
    local values:            local cell values as a scalar PCRaster field;
    zones:                   zones over which the minimum values are computed as
                             a nominal PCRaster field, or None if every cell
                             is its own zone.

    Output:
    =======
    minimums:                minimum values over the zones per cell as a scalar
                             PCRaster field.

"""
    # without zones, every cell is its own zone
    if zones is None:
        return local_values
    return pcr.areaminimum(local_values, zones)


def get_zonal_fraction(local_values: pcr.Field, zones: pcr.Field | None) -> pcr.Field:
    """
get_zonal_fraction: function that computes the fractional value per cell over \
the total of the provided zones.

    Input:
    ======
    local values:            local cell values as a scalar PCRaster field;
    zones:                   zones over which the fractional values for the
                             cells are computed as a nominal PCRaster field,
                             or None if every cell is its own zone (the
                             fraction is then 1 where the local value is
                             positive).

    Output:
    =======
    fractional_values:       fractional values, summing to unity over the ap-
                             propriate zone, as a scalar PCRaster field.

"""

    totals = get_zonal_total(local_values, zones)

    fractional_values = getValDivZero(local_values, totals, very_small_number)

    return fractional_values


_zonal_cellarea_cache: dict[int, tuple[pcr.Field, pcr.Field]] = {}


def water_balance_check(
    states_ini,
    states_end,
    cellarea,
    process_name,
    var_name,
    date,
    zones=None,
    flag_warning=True,
    flag_debug=False,
    threshold=1e-5,
):
    """
    water_balance_check :
                   function to evaluate the water balance for a list of
                   input and output map files

    input:
    =====
    states_ini   : list of variable names of the initial state of variables
                   before processing, to be aggregated (units: m3/day)
    states_end   : list of variable names of the final state of variables
                   after processing, to be aggregated (units: m3/day)
    cellarea     : PCRaster map with cell area (units: m2)
    process_name : string with the name of the process evaluated
    date         : string with the current date
    zones        : PCRaster map with allocation zones per water source used to
                   aggregate water volumes
    flag_warning : boolean to specify if QUAlloc must get halted if water balance
                   does not close (False) or only print a warning (True, default)
    flag_debug   : boolean to specify if map is reported (True) in case the water
                   balance does not close or not (False, default)
    threshold    : float, minimum value allowed to determined if the water balance
                   is closed (units: m water-slice)
    """

    in_map = pcr.spatial(pcr.scalar(0.0))
    out_map = pcr.spatial(pcr.scalar(0.0))

    # aggregate all water on each side of the process
    for state_ini in states_ini:
        in_map += state_ini
    for state_end in states_end:
        out_map += state_end

    # aggregate the water volumes over the allocation zones
    if not isinstance(zones, NoneType):
        in_out_map = get_zonal_total(in_map - out_map, zones)
        cached = _zonal_cellarea_cache.get(id(zones))
        if cached is None or cached[0] is not zones:
            cached = (zones, get_zonal_total(cellarea, zones))
            _zonal_cellarea_cache[id(zones)] = cached
        zonal_cellarea = cached[1]
    else:
        in_out_map = in_map - out_map
        zonal_cellarea = cellarea

    # difference as water slice
    diff = in_out_map / zonal_cellarea
    vmin = pcr.cellvalue(pcr.mapminimum(diff), 1)[0]

    if vmin >= -threshold:
        msg = f"[ {date!s:.10} Water Balance ] {process_name} ({var_name}): OK"
        if vmin < 0:
            msg += f" (max mismatch of {vmin:.2e} m)"
        logger.debug(msg)

    else:
        bar = "#" * 161
        logger.error(
            f"\n{bar}\nWARNING !!!!!!!! [ Water Balance Error ] {process_name} ({var_name}): "
            f"max mismatch of {vmin:.10f} m [ {date} ]\n{bar}\n"
        )

        if flag_debug:
            pcr.aguila(diff, diff * zonal_cellarea)

        if not flag_warning:
            raise RuntimeError(msg)


def ncvariable_name(section, key, default=None):
    """
ncvariable_name: returns the name of the netCDF variable that holds the entry \
of key in the given configuration section. This is the key itself, or default \
where given, unless the section provides an explicit <key>_ncvariable override; \
PCRaster input ignores the name and is unaffected.
"""

    if isinstance(default, NoneType):
        default = key

    return section.get(f"{key}_ncvariable", default)


def modify_ini_file(original_ini_file, system_argument):

    # created by Edwin H. Sutanudjaja in August 2020 for the Ulysses project

    file_ini = open(original_ini_file, "rt")
    file_ini_content = file_ini.read()
    file_ini.close()

    # output directory (-mod); always required
    main_output_dir = system_argument[system_argument.index("-mod") + 1]
    file_ini_content = file_ini_content.replace("MAIN_OUTPUT_DIR", main_output_dir)
    msg = (
        "The output folder 'outputDir' is set based on the system argument (-mod): "
        + main_output_dir
    )
    print(msg)

    # input directory (-mid)
    if "-mid" in system_argument:
        main_input_dir = system_argument[system_argument.index("-mid") + 1]
        file_ini_content = file_ini_content.replace("MAIN_INPUT_DIR", main_input_dir)
        msg = (
            "The input folder 'inputDir' is set based on the system argument (-mid): "
            + main_input_dir
        )
        print(msg)

    # optional start (-sd) and end (-ed) dates
    if "-sd" in system_argument:
        starting_date = system_argument[system_argument.index("-sd") + 1]
        file_ini_content = file_ini_content.replace("START_DATE", starting_date)
        msg = (
            "The starting date 'startTime' is set based on the system argument (-sd): "
            + starting_date
        )
        print(msg)
    if "-ed" in system_argument:
        end_date = system_argument[system_argument.index("-ed") + 1]
        file_ini_content = file_ini_content.replace("END_DATE", end_date)
        msg = (
            "The end date 'END_DATE' is set based on the system argument (-ed): "
            + end_date
        )
        print(msg)
    for token, flag in [("START_DATE", "-sd"), ("END_DATE", "-ed")]:
        if token in file_ini_content:
            sys.exit(
                f"{original_ini_file} uses the placeholder {token}; pass it with {flag}"
            )

    # optional initial conditions: main initial state folder (-misd)
    if "-misd" in system_argument:
        main_initial_state_folder = system_argument[system_argument.index("-misd") + 1]
        file_ini_content = file_ini_content.replace(
            "MAIN_INITIAL_STATE_FOLDER", main_initial_state_folder
        )
        msg = (
            "The main folder for all initial states is set based on the system argument (-misd): "
            + main_initial_state_folder
        )
        print(msg)
    # date for initial states (-dfis)
    if "-dfis" in system_argument:
        date_for_initial_states = system_argument[system_argument.index("-dfis") + 1]
        file_ini_content = file_ini_content.replace(
            "DATE_FOR_INITIAL_STATES", date_for_initial_states
        )
        msg = (
            "The date for all initial state files is set based on the system argument (-dfis): "
            + date_for_initial_states
        )
        print(msg)

    # DOMAIN_CONNECTIONS
    if "-clone_connections" in system_argument:
        clone_connections = system_argument[
            system_argument.index("-clone_connections") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "CLONE_CONNECTIONS", clone_connections
        )
        msg = (
            "The clone_connections is set based on the system argument (-clone_connections): "
            + clone_connections
        )
        print(msg)

    # optional forcing files; precipitationNC = PRECIPITATION_FORCING_FILE
    if "-pff" in system_argument:
        precipitation_forcing_file = system_argument[system_argument.index("-pff") + 1]
        file_ini_content = file_ini_content.replace(
            "PRECIPITATION_FORCING_FILE", precipitation_forcing_file
        )
        msg = (
            "The precipitation forcing file 'precipitationNC' is set based on the system argument (-pff): "
            + precipitation_forcing_file
        )
        print(msg)

    # temperatureNC = TEMPERATURE_FORCING_FILE
    if "-tff" in system_argument:
        temperature_forcing_file = system_argument[system_argument.index("-tff") + 1]
        file_ini_content = file_ini_content.replace(
            "TEMPERATURE_FORCING_FILE", temperature_forcing_file
        )
        msg = (
            "The temperature forcing file 'temperatureNC' is set based on the system argument (-tff): "
            + temperature_forcing_file
        )
        print(msg)

    # refETPotFileNC = REF_POT_ET_FORCING_FILE
    if "-rpetff" in system_argument:
        ref_pot_et_forcing_file = system_argument[system_argument.index("-rpetff") + 1]
        file_ini_content = file_ini_content.replace(
            "REF_POT_ET_FORCING_FILE", ref_pot_et_forcing_file
        )
        msg = (
            "The reference potential ET forcing file 'refETPotFileNC' is set based on the system argument (-tff): "
            + ref_pot_et_forcing_file
        )
        print(msg)

    # atmospheric_pressure = PRESSURE_FORCING_FILE
    if "-presff" in system_argument:
        pressure_forcing_file = system_argument[system_argument.index("-presff") + 1]
        file_ini_content = file_ini_content.replace(
            "PRESSURE_FORCING_FILE", pressure_forcing_file
        )
        msg = (
            "The pressure forcing file 'atmospheric_pressure' is set based on the system argument (-presff): "
            + pressure_forcing_file
        )
        print(msg)

    # wind_speed_10m = WIND_FORCING_FILE
    if "-windff" in system_argument:
        wind_forcing_file = system_argument[system_argument.index("-windff") + 1]
        file_ini_content = file_ini_content.replace(
            "WIND_FORCING_FILE", wind_forcing_file
        )
        msg = (
            "The wind forcing file 'wind_speed_10m' is set based on the system argument (-windff): "
            + wind_forcing_file
        )
        print(msg)

    # shortwave_radiation = SHORTWAVE_RADIATION_FORCING_FILE
    if "-swradff" in system_argument:
        shorwave_radiation_forcing_file = system_argument[
            system_argument.index("-swradff") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "SHORTWAVE_RADIATION_FORCING_FILE", shorwave_radiation_forcing_file
        )
        msg = (
            "The shortwave radiation forcing file 'shortwave_radiation' is set based on the system argument (-windff): "
            + shorwave_radiation_forcing_file
        )
        print(msg)

    # relative_humidity = RELATIVE_HUMIDITY_FORCING_FILE
    if "-relhumff" in system_argument:
        relative_humidity_forcing_file = system_argument[
            system_argument.index("-relhumff") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "RELATIVE_HUMIDITY_FORCING_FILE", relative_humidity_forcing_file
        )
        msg = (
            "The relative humidity forcing file 'relative_humidity' is set based on the system argument (-relhumff): "
            + relative_humidity_forcing_file
        )
        print(msg)

    # optional baseflow exponent
    if "-bfexp" in system_argument:
        baseflow_exponent = system_argument[system_argument.index("-bfexp") + 1]
        file_ini_content = file_ini_content.replace(
            "BASEFLOW_EXP_INPUT", baseflow_exponent
        )
        msg = (
            "The groundwater baseflow exponent 'bfexp' is set based on the system argument (-bfexp): "
            + baseflow_exponent
        )
        print(msg)

    # NUMBER_OF_SPINUP_YEARS
    if "-num_of_sp_years" in system_argument:
        number_of_spinup_years = system_argument[
            system_argument.index("-num_of_sp_years") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "NUMBER_OF_SPINUP_YEARS", number_of_spinup_years
        )
        msg = (
            "The number_of_spinup_years is set based on the system argument (-num_of_sp_years): "
            + number_of_spinup_years
        )
        print(msg)

    # CLONEMAP
    if "-clonemap" in system_argument:
        clonemap = system_argument[system_argument.index("-clonemap") + 1]
        file_ini_content = file_ini_content.replace("CLONEMAP", clonemap)
        msg = (
            f"The clonemap is set based on the system argument (-clonemap): {clonemap}"
        )
        print(msg)

    # USE_MAXIMUM_STOR_GROUNDWATER_FOSSIL_INI
    if "-use_max_fossil_gw_ini" in system_argument:
        use_max_fossil_gw_ini = system_argument[
            system_argument.index("-use_max_fossil_gw_ini") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "USE_MAXIMUM_STOR_GROUNDWATER_FOSSIL_INI", use_max_fossil_gw_ini
        )
        msg = (
            "The option 'useMaximumStorGroundwaterFossilIni' is set based on the system argument (-use_max_fossil_gw_ini): "
            + use_max_fossil_gw_ini
        )
        print(msg)

    # ESTIMATE_STOR_GROUNDWATER_INI_FROM_RECHARGE
    if "-est_stor_gw_from_rch" in system_argument:
        est_stor_gw_from_rch = system_argument[
            system_argument.index("-est_stor_gw_from_rch") + 1
        ]
        file_ini_content = file_ini_content.replace(
            "ESTIMATE_STOR_GROUNDWATER_INI_FROM_RECHARGE", est_stor_gw_from_rch
        )
        msg = (
            "The option 'estimateStorGroundwaterIniFromRecharge' is set based on the system argument (-est_stor_gw_from_rch): "
            + est_stor_gw_from_rch
        )
        print(msg)

    # dailyGroundwaterRechargeIni / DAILY_GROUNDWATER_RECHARGE_INI
    if "-day_gw_rch_ini" in system_argument:
        day_gw_rch_ini = system_argument[system_argument.index("-day_gw_rch_ini") + 1]
        file_ini_content = file_ini_content.replace(
            "DAILY_GROUNDWATER_RECHARGE_INI", day_gw_rch_ini
        )
        msg = (
            "The option 'dailyGroundwaterRechargeIni' is set based on the system argument (-day_gw_rch_ini): "
            + day_gw_rch_ini
        )
        print(msg)

    # configuration_file_for_qualloc / QUALLOC_CONFIG_FILE
    if "-qcf" in system_argument:
        qualloc_config_file = system_argument[system_argument.index("-qcf") + 1]
        file_ini_content = file_ini_content.replace(
            "QUALLOC_CONFIG_FILE", qualloc_config_file
        )
        msg = (
            "The configuration file to run QUAlloc is set based on the system argument (-qcf): "
            + qualloc_config_file
        )
        print(msg)

    # folder for the original and modified ini files
    folder_for_ini_files = os.path.join(main_output_dir, "ini_files")
    # for a run that is part of a set of parallel (clone) runs
    if (
        system_argument[2] == "parallel"
        or system_argument[2] == "debug_parallel"
        or system_argument[2] == "debug-parallel"
    ):
        clone_code = str(system_argument[3])
        output_folder_with_clone_code = f"M{int(clone_code):02}"
        folder_for_ini_files = os.path.join(
            main_output_dir, output_folder_with_clone_code, "ini_files"
        )

    if os.path.exists(folder_for_ini_files):
        shutil.rmtree(folder_for_ini_files)
    os.makedirs(folder_for_ini_files)

    shutil.copy(
        original_ini_file,
        os.path.join(
            folder_for_ini_files, f"{os.path.basename(original_ini_file)}.original"
        ),
    )

    new_ini_file_name = os.path.join(
        folder_for_ini_files, f"{os.path.basename(original_ini_file)}.modified_and_used"
    )
    new_ini_file = open(new_ini_file_name, "w")
    new_ini_file.write(file_ini_content)
    new_ini_file.close()

    return new_ini_file_name
