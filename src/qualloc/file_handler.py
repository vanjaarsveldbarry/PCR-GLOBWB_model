import logging
import os
import shutil
import tempfile

import pcraster as pcr

from common import virtualOS as vos

logger = logging.getLogger(__name__)


# global parameters: missing value identifier, types, file extensions
missing_value = -999.9
very_small_number = 1.0e-12

NoneType = type(None)

# default file extensions
file_extensions = {
    ".map": "pcraster",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".tif": "pcraster",
    "": "",
}

# file conversion settings

conversion_methods = {
    "Scalar": float,
    "Nominal": int,
    "Boolean": bool,
    "Ordinal": int,
    "Directional": float,
    "Ldd": int,
}

# netCDF files opened for QUAlloc; they share vos.filecache with PCR-GLOBWB
nc_files_read = set()


# generic functions to process files


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


def read_file_entry(
    filename,
    variablename,
    inputpath="",
    file_subst_args=(),
    forced_non_spatial=False,
    datatype=pcr.Scalar,
    date=None,
    date_selection_method="exact",
    allow_year_substitution=False,
):
    """

read_file_entry: generic function that can read information from file for a \
given date. This may concern spatial information or single entries.

    Input:
    ======
    required input:
    ---------------
    filename:               file name or root of the file to be extracted,
                            this can refer to PCRaster, netCDF files or a
                            value as a string;
    variablename:           name of the variable to be extracted, read from
                            the specified file;

    optional input:         defaults are None unless specified otherwise;
    ---------------
    inputpath:              input path that is attached to the file name if the
                            latter is a relative path (default: '');
    file_subst_args:        file substitution arguments that are used to repl-
                            enish the file name if needed (default: ());
    forced_non_spatial:     boolean forcing multidimensional data to be read
                            as non-spatial, applies to netCDF files only or in
                            case of the conversion of numerical values
                            (default: False);
    datatype:               PCRaster data type to be extracted
                            (default: pcr.Scalar);
    date:                   date to be extracted;
    date_selection_method:  option to select alternative dates if the actual
                            date is not met (default: 'exact');
    allow_year_substitution:
                            for temporal netCDF data, allows dates to be
                            changed and match a particular year; this is part-
                            icularly useful to read climatologies or reuse
                            existing data over longer periods (e.g., spinup),
                            (default: False).

    Output:
    =======
    var_out:                output for the variable, either spatial or
                            non-spatial.

"""
    var_out = None

    filename = str(filename)
    val_str = str(filename)
    datatype_str = str(datatype)
    if "VALUESCALE." in datatype_str:
        datatype_str = datatype_str.replace("VALUESCALE.", "")

    # compose the file name and check that it exists
    filename, existing_file = compose_filename(filename, inputpath, file_subst_args)

    if existing_file and file_is_nc(filename):

        nc_files_read.add(filename)
        if vos._hasTimeAxis(filename, variablename):
            # QUAlloc only substitutes years when before/after finds no date at all;
            # its one before/after caller (initial conditions) always asks for a date
            # that has one, so substitution is only passed on with exact
            var_out = vos.netcdf2PCRobjClone(
                filename,
                variablename,
                dateInput=date,
                valueScale=datatype,
                select=date_selection_method.lower(),
                yearSubstitution=bool(allow_year_substitution)
                and date_selection_method.lower() == "exact",
            )
        else:
            var_out = vos.netcdf2PCRobjCloneWithoutTime(
                filename, variablename, valueScale=datatype
            )

    elif existing_file and file_is_pcr(filename):

        # readPCRmapClone empties and recreates tmpDir, so it gets one of its own
        tmp_dir = tempfile.mkdtemp(prefix="qualloc_") + os.sep
        try:
            var_out = vos.readPCRmapClone(filename, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        var_out = getattr(pcr, datatype_str.lower())(var_out)

        # output available: log the message
        message_str = f"value of {variablename} read from {filename}"
        message_str = str.join(
            " ",
            (message_str, "in PCRaster format which does not contain variable info"),
        )
        if not isinstance(date, NoneType):
            message_str = str.join(
                " ", (message_str, f"and is unaware of the actual date {date}")
            )

        logger.debug(message_str)

    else:

        # the file type cannot be read
        try:
            var_out = float(val_str)
            if forced_non_spatial:
                conversion_method = conversion_methods[datatype_str.title()]
            else:
                conversion_method = getattr(pcr, datatype_str.lower())
            var_out = conversion_method(var_out)
            logger.debug("%s is a value; converted to %s", val_str, conversion_method)
        except Exception:
            logger.error("%s: not a netCDF/PCRaster file, cannot convert", filename)

    return var_out


def close_nc_cache():
    """closes the netCDF input files opened for QUAlloc"""

    for filename in nc_files_read:
        f = vos.filecache.pop(filename, None)
        if f is not None and f.isopen():
            f.close()
        vos._gridcache.pop(filename, None)
    nc_files_read.clear()

    return None


def main():
    pass


if __name__ == "__main__":
    main()
