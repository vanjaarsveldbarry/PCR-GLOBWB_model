import logging
import sys
from copy import deepcopy

import netCDF4 as nc
import numpy as np
import pcraster as pcr

logger = logging.getLogger(__name__)

NoneType = type(None)
DictType = type(dict)

# default netCDF format and variable attributes for createVariable
default_nc_format = "NETCDF3_CLASSIC"
default_zlib = False
default_complevel = 4
default_shuffle = True
default_fletcher32 = False
default_contiguous = False
default_chunksizes = None
default_endian = "native"
default_least_significant_digit = None
default_fill_value = -999.9


def initialize_ncfile(
    ncfilename,
    nc_format=None,
    nc_global_attributes={},
    cache={},
):
    """

initialize_netCD: initializes the netCDF file with the name and format specified\
     and set its attributes for output.

"""

    if isinstance(nc_format, NoneType):
        nc_format = default_nc_format

    if ncfilename in cache:
        rootgrp = cache[ncfilename]
    else:
        rootgrp = nc.Dataset(ncfilename, "w", format=nc_format)

    for attribute, value in nc_global_attributes.items():
        rootgrp.setncattr(attribute, value)

    # close the file if it is not in the cache
    if ncfilename not in cache:
        rootgrp.close()

    return None


def add_variable_to_netCDF(
    ncfilename,
    name,
    datatype,
    dimensions,
    cache={},
    **nc_optional_info,
):
    """

add_variable_to_netCDF: function that adds a variable to the netCDF file using\
the netCDF4 createVariable function.

    Input:
    ======
    required inputs:
    ----------------
    ncFile:             the file name of the netCDF dataset in which the variable will be created
    name:               the name of the variable by which it is stored
    datatype:           the data type, numpy datatype object or its dtype.str attribute
                        or any of the supported specifiers included: 'S1' or 'c' (NC_CHAR),
                        'i1' or 'b' or 'B' (NC_BYTE), 'u1' (NC_UBYTE), 'i2' or
                        'h' or 's' (NC_SHORT), 'u2' (NC_USHORT), 'i4' or 'i' or
                        'l' (NC_INT), 'u4' (NC_UINT), 'i8' (NC_INT64), 'u8' (NC_UINT64),
                        'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE).
    dimensions:         tuple with dimensions.

    optional inputs:
    ----------------
    cache:              dictionary with already open files.

    Any or all of the following:
    zlib, complevel, shuffle,
    fletcher32, contiguous, chunksizes, endian,
    least_significant_digit, fill_value;
    nc_var_attrs:       a dictionary holding all netCDF variable attributes;
                        if this dictionary holds any of the above parameters but it is
                        specified explicitly, the latter is given precedence. Otherwise it is
                        set implicitly using the default, global values.

    """

    # process the optional keywords and set the defaults
    nc_defaults = {}
    nc_defaults["varname"] = name
    nc_defaults["datatype"] = datatype
    nc_defaults["dimensions"] = dimensions
    nc_defaults["zlib"] = default_zlib
    nc_defaults["complevel"] = default_complevel
    nc_defaults["shuffle"] = default_shuffle
    nc_defaults["fletcher32"] = default_fletcher32
    nc_defaults["contiguous"] = default_contiguous
    nc_defaults["chunksizes"] = default_chunksizes
    nc_defaults["endian"] = default_endian
    nc_defaults["fill_value"] = default_fill_value
    nc_defaults["least_significant_digit"] = default_least_significant_digit

    # optional nc_var_attrs
    if "nc_var_atttrs" in nc_optional_info.keys() and isinstance(
        nc_optional_info["nc_var_attrs"], DictType
    ):
        nc_var_attrs = nc_optional_info["nc_var_attrs"].copy()
    else:
        nc_var_attrs = {}

    # pass keywords to the defaults
    for key, value in nc_optional_info.items():

        if key != "nc_var_attrs":

            if key not in nc_var_attrs.keys():

                nc_var_attrs[key] = value

    # update the defaults
    nc_var_keys = list(nc_var_attrs.keys())
    for key in nc_var_keys:

        if key in nc_defaults.keys():

            nc_var_attrs[key] = nc_var_attrs[key]

            # strip the keyword from the attributes
            del nc_var_attrs[key]

    if ncfilename in cache:
        rootgrp = cache[ncfilename]
    else:
        rootgrp = nc.Dataset(ncfilename, "a")

    # create the variable, first with the defaults, then with the remaining keys of nc_var_attrs
    nc_variable = rootgrp.createVariable(**nc_defaults)

    if "ncattrs" in nc_var_attrs.keys():

        for key, value in nc_var_attrs["nc_attrs"].items():

            nc_variable.setncattr(key, value)

    else:

        # remaining attributes
        for key, value in nc_var_attrs.items():

            nc_variable.setncattr(key, value)

    rootgrp.sync()

    # close the file if it is not in the cache
    if ncfilename not in cache:
        rootgrp.close()

    return None


def add_dimension_to_netCDF(
    ncfilename,
    name,
    datatype,
    unlimited=False,
    values=[],
    cache={},
    **nc_optional_info,
):
    """

add_dimension_to_netCDF: function that adds a dimension and the associated \
variable information to the netCDF file.

    Input:
    ======
    required inputs:
    ----------------
    ncFile:             the file name of the netCDF dataset in which the variable will be created
    name:               the name of the dimension that is created
    datatype:           the data type, numpy datatype object or its dtype.str attribute
                        or any of the supported specifiers included: 'S1' or 'c' (NC_CHAR),
                        'i1' or 'b' or 'B' (NC_BYTE), 'u1' (NC_UBYTE), 'i2' or
                        'h' or 's' (NC_SHORT), 'u2' (NC_USHORT), 'i4' or 'i' or
                        'l' (NC_INT), 'u4' (NC_UINT), 'i8' (NC_INT64), 'u8' (NC_UINT64),
                        'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE).

    optional inputs:
    ----------------
    unlimited:          boolean specifying whether the dimension is unlimited or not;
    values:             if not unlimited, then provide the values that characterize
                        the dimension;
    cache:              dictionary with already open files.

    Any or all of the following to create the variables:
    zlib, complevel, shuffle,
    fletcher32, contiguous, chunksizes, endian,
    least_significant_digit, fill_value;
    nc_var_attrs:       a dictionary holding all netCDF variable attributes;
                        if this dictionary holds any of the above parameters but it is
                        specified explicitly, the latter is given precedence. Otherwise it is
                        set implicitly using the default, global values.

    """

    if ncfilename in cache:
        rootgrp = cache[ncfilename]
    else:
        rootgrp = nc.Dataset(ncfilename, "a")

    if unlimited:
        size = None
    else:
        size = len(values)

    rootgrp.createDimension(name, size)

    rootgrp.sync()

    # close the file if it is not in the cache
    if ncfilename not in cache:
        rootgrp.close()

    add_variable_to_netCDF(
        ncfilename=ncfilename,
        name=name,
        datatype=datatype,
        dimensions=(name,),
        cache=cache,
        **nc_optional_info,
    )

    if not isinstance(size, NoneType):

        if ncfilename in cache:
            rootgrp = cache[ncfilename]
        else:
            rootgrp = nc.Dataset(ncfilename, "a")

        rootgrp.variables[name][:] = values[:]

        rootgrp.sync()

        # close the file if it is not in the cache
        if ncfilename not in cache:
            rootgrp.close()

    return None


def add_data_to_netCDF(
    ncfilename, name, variable_array, dim_slices, cache={}, **additional_info
):
    """

    add_data_to_netCDF: function that adds data to the netCDF file specified.

        Input:
        ======
        required inputs:
        ----------------
        ncfilename:         the file name of the netCDF dataset in which the variable will be created;
        name:               the name of the variable that is updated;
        variable_array:     the array with that will be added to the netCDF;
        dim_slices:         slice of dimensions: dictionary with the location on where
                            the variable field should be inserted; default value
                            is None in which case the value is inserted at the end/
                            over the full domain. If specified, the values should be
                            a tuple of the first and last indices of the slice;
        cache:              dictionary with already open files;
        additional_info:    additional information; this should include a
                            list of dates (dates) as well as the name of the
                            dimension that holds the time stamps (time_dimension)
                            to add temporal data.

        Output:
        =======
        None:               returns None.

    """

    if ncfilename in cache:
        rootgrp = cache[ncfilename]
    else:
        rootgrp = nc.Dataset(ncfilename, "a")

    # copy the dimension slices and the variable array
    dim_slices = deepcopy(dim_slices)
    variable_array = deepcopy(variable_array)

    simple_update = True

    dim_keys = list(rootgrp.variables[name].dimensions)

    time_dimension = ""
    if "time_dimension" in additional_info.keys() and "dates" in additional_info.keys():

        time_dimension = additional_info["time_dimension"]
        dates = additional_info["dates"]

        # temporal information: add the dates
        nc_time = rootgrp.variables[time_dimension]
        if len(nc_time[:]) > 0:
            try:
                date_ixs = np.array(nc.date2index(dates, nc_time)).ravel()
            except Exception:
                date_ixs = np.arange(len(dates)) + len(nc_time[:])
        else:
            date_ixs = np.arange(len(dates)) + len(nc_time[:])

        date_ixs = date_ixs.tolist()
        for date_ix in date_ixs:

            date = dates[date_ixs.index(date_ix)]
            nc_time[date_ix] = nc.date2num(date, nc_time.units, nc_time.calendar)

        # insert the date indices if the dimension values are not set
        if isinstance(dim_slices[time_dimension], NoneType):

            simple_update = True

            # first and last position
            dim_slices[time_dimension] = (date_ixs[0], date_ixs[-1] + 1)

        else:
            simple_update = False

    # set dimension slices that are None, and get the required size of the (sliced) variable array;
    # processing is simplified for the maximum non-temporal dimensions
    req_size = 1

    for dim_key in dim_keys:

        if isinstance(dim_slices[dim_key], NoneType):

            dim_slices[dim_key] = (0, len(rootgrp.variables[dim_key][:]))

        if dim_key != time_dimension:
            simple_update = simple_update and (
                (dim_slices[dim_key][1] - dim_slices[dim_key][0])
                == len(rootgrp.variables[dim_key][:])
            )

        req_size = req_size * (dim_slices[dim_key][1] - dim_slices[dim_key][0])

    while variable_array.size != req_size:
        sys.exit("array sizes do not match!")

    # a simple update is possible if the variable array has the size of a single time step

    if simple_update:

        if time_dimension != "":
            # timed: write to the last field
            rootgrp.variables[name][dim_slices[time_dimension][0], ...] = (
                variable_array[:]
            )
        else:
            # not timed: write the full array
            rootgrp.variables[name][...] = variable_array[:]

    else:

        # otherwise, use a boolean mask and index array to set the corresponding entries

        mask_array = np.ones(rootgrp.variables[name][:].shape, dtype=bool)
        g_ixs = np.indices(mask_array.shape)
        for dim_key in dim_keys:

            dim_ix = dim_keys.index(dim_key)

            mask_ix = (g_ixs[dim_ix] >= dim_slices[dim_key][0]) & (
                g_ixs[dim_ix] < dim_slices[dim_key][1]
            )

            mask_array = mask_array & mask_ix

        v_a = rootgrp.variables[name][:].copy()

        v_a[mask_array] = variable_array[:].ravel()

        rootgrp.variables[name][:] = v_a.copy()

    rootgrp.sync()

    v_a = None
    mask_array = None
    mask_ix = None
    del v_a, mask_array, mask_ix

    # close the file if it is not in the cache
    if ncfilename not in cache:
        rootgrp.close()

    return None


class netCDF_output_handler(object):
    """

netCDF_output_handler: class that holds all information to create and update \
output netCDF files.

"""

    def __init__(self, model_configuration):

        object.__init__(self)

        # cache of open files, dimensions per variable and the timed variable
        self.cache = dict()
        self.dimensions = dict()
        self.time_dimension = dict()

        self.latitude = pcr.pcr2numpy(
            pcr.ycoordinate(pcr.spatial(pcr.boolean(1))), default_fill_value
        )[:, 0]
        self.longitude = pcr.pcr2numpy(
            pcr.xcoordinate(pcr.spatial(pcr.boolean(1))), default_fill_value
        )[0, :]

        # netCDF format and zlib setup
        self.default_fill_value = default_fill_value
        self.nc_format = default_nc_format
        self.zlib = default_zlib
        if "nc_format" in model_configuration.reporting.keys():
            self.format = model_configuration.convert_string_to_input(
                model_configuration.reporting["nc_format"], str
            )
        if "zlib" in model_configuration.reporting.keys():
            self.format = model_configuration.convert_string_to_input(
                model_configuration.reporting["zlib"], bool
            )

        # general netCDF attributes from the ini file
        self.nc_global_attributes = self.get_general_netcdf_attributes(
            model_configuration
        )

        self.dimension_info = {}

        # dimension info for time and the x and y coordinates
        self.dimension_info["time"] = {
            "dim_pos": 0,
            "is_temporal": True,
            "is_spatial": False,
            "datatype": "f4",
            "unlimited": True,
            "long_name": "Days since 1901-01-01",
            "standard_name": "Days since 1901-01-01",
            "calendar": "standard",
            "units": "Days since 1901-01-01",
        }
        self.dimension_info["latitude"] = {
            "dim_pos": 1,
            "is_temporal": False,
            "is_spatial": True,
            "datatype": "f4",
            "unlimited": False,
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
        self.dimension_info["longitude"] = {
            "dim_pos": 2,
            "is_temporal": False,
            "is_spatial": True,
            "datatype": "f4",
            "unlimited": False,
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }

        return None

    def get_general_netcdf_attributes(self, model_configuration):

        nc_default_attributes = {
            "title": f"QUAlloc run for {model_configuration.general['scenarioname']}",
            "history": f"Created on {model_configuration._timestamp_str}",
            "institution": "Dept. Physical Geography, Utrecht University (r.vanbeek@uu.nl)",
        }
        nc_global_attributes = {}

        # add all attributes from the dictionary
        for key, attribute in model_configuration.netcdfattrs.items():

            nc_global_attributes[key] = attribute

        # add defaults if necessary
        for key, attribute in nc_default_attributes.items():

            if key not in nc_global_attributes.keys():
                nc_global_attributes[key] = attribute

        return nc_global_attributes

    def test_ncfile_in_cache(self, ncfilename):
        """tests if the specified nc file name is present in the cache"""

        return ncfilename in self.cache.keys()

    def add_ncfile_to_cache(
        self,
        ncfilename,
    ):
        """adds the information from the specified netCDF file to this instance \
holding netCDF information to facilitate access."""

        # add the netCDF file to the cache if not present yet
        if not self.test_ncfile_in_cache(ncfilename):

            self.cache[ncfilename] = nc.Dataset(ncfilename, "w", format=self.nc_format)

            logger.info(f"neCDF file {ncfilename} added to cache")

    def remove_ncfile_from_cache(self, ncfilename):
        """closes the netCDF file and remove all information from the \
specified netCDF file."""

        # remove the netCDF file from the cache if present
        if ncfilename in self.cache.keys():

            self.cache[ncfilename].close()
            del self.cache[ncfilename]

            logger.info(f"neCDF file {ncfilename} removed from cache")

    def initialize_ncfile(self, ncfilename):
        """
initialize_ncfile: function of the netCDF_output_handler that wraps around \
initialize_ncfile to create the output netCDF file and adds it to the cache \
for reduced IO.

"""

        self.add_ncfile_to_cache(ncfilename)

        initialize_ncfile(
            ncfilename=ncfilename,
            nc_format=self.nc_format,
            nc_global_attributes=self.nc_global_attributes,
            cache=self.cache,
        )

    def initialize_nc_variable(
        self,
        ncfilename,
        variablename,
        variable_units,
        is_spatial,
        is_temporal,
        long_name,
        standard_name,
        datatype,
    ):
        """

initialize_nc_variable: function of the netCDF_output_handler that creates the \
variable and adds all the necessary dimensions to the output netCDF file and \
includes its dimensions and time dimension to the cache when appropriate.
Function wraps around the functions add_dimension_to_netCDF and add_variable_to_netCDF.

"""

        # initialize the netCDF file if necessary
        if not self.test_ncfile_in_cache(ncfilename):
            self.initialize_ncfile(ncfilename)

        var_dim_keys = []
        for dim_key, dim_info in self.dimension_info.items():
            dim_pos = dim_info["dim_pos"]
            if dim_info["is_spatial"] and is_spatial:
                var_dim_keys.insert(dim_pos, dim_key)
            if dim_info["is_temporal"] and is_temporal:
                var_dim_keys.insert(dim_pos, dim_key)

        nc_dim_keys = list(self.cache[ncfilename].dimensions.keys())

        for dim_key in var_dim_keys:
            if dim_key not in nc_dim_keys:

                logger.debug(f"dimension {dim_key} added to {ncfilename}")

                dim_info = self.dimension_info[dim_key]

                # temporal
                if dim_info["is_temporal"]:

                    add_dimension_to_netCDF(
                        ncfilename=ncfilename,
                        name=dim_key,
                        datatype=datatype,
                        unlimited=dim_info["unlimited"],
                        cache=self.cache,
                        long_name=dim_info["long_name"],
                        standard_name=dim_info["standard_name"],
                        calendar=dim_info["calendar"],
                        units=dim_info["units"],
                    )

                    # add the temporal variable
                    self.time_dimension[ncfilename] = dim_key

                # spatial
                elif dim_info["is_spatial"]:

                    values = getattr(self, dim_key)
                    add_dimension_to_netCDF(
                        ncfilename=ncfilename,
                        name=dim_key,
                        datatype=datatype,
                        unlimited=dim_info["unlimited"],
                        values=values,
                        cache=self.cache,
                        long_name=dim_info["long_name"],
                        standard_name=dim_info["standard_name"],
                        units=dim_info["units"],
                    )

                else:
                    pass

        # dimensions of the current variable
        if ncfilename not in self.dimensions.keys():
            self.dimensions[ncfilename] = {variablename: var_dim_keys}

        logger.debug(f"variable {variablename} added to {ncfilename}")

        add_variable_to_netCDF(
            ncfilename=ncfilename,
            name=variablename,
            datatype=datatype,
            dimensions=var_dim_keys,
            cache=self.cache,
            long_name=long_name,
            standard_name=standard_name,
            units=variable_units,
        )

        return None

    def add_data_to_netCDF(
        self, ncfilename, variablename, variable_array, **additional_info
    ):
        """

add_data_to_netCDF: function of the netCDF_output_handler that adds the output \
to netCDF file for the variable specified.
Function wraps around the function add_data_to_netCDF and initializes the tuple \
of the dimensions to be written.

"""

        # time variable and date of temporal variables
        is_timed = False

        if "is_timed" in additional_info.keys():
            is_timed = additional_info["is_timed"]

        if is_timed:
            if "time_dimension" not in additional_info.keys():
                additional_info["time_dimension"] = self.time_dimension[ncfilename]

        dim_slices = dict(
            [(dim_key, None) for dim_key in self.dimensions[ncfilename][variablename]]
        )

        add_data_to_netCDF(
            ncfilename=ncfilename,
            name=variablename,
            variable_array=variable_array,
            dim_slices=dim_slices,
            cache=self.cache,
            **additional_info,
        )

        return None

    def close_cache(self):

        for ncfilename in list(self.cache.keys()):

            self.remove_ncfile_from_cache(ncfilename)

        return None
