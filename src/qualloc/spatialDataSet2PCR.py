import os
import subprocess
import sys
import types

import pcraster as pcr

NoneType = type(None)


class spatialAttributes:
    # attributes of a spatial dataset

    def __init__(self, inputFileName, maxLength=-1):
        """Returns the map attributes for the spatial file name specified"""
        # maxLength: cutoff to speed up processing of large datasets with multiple bands;
        # mapInformation holds the identifier strings and offsets of the variables of interest
        mapInformation = {}
        mapInformation["dataFormat"] = "Driver", 0, str
        mapInformation["numberRows"] = "Size is", 1, int
        mapInformation["numberCols"] = "Size is", 0, int
        mapInformation["xResolution"] = "Pixel Size =", 0, float
        mapInformation["yResolution"] = "Pixel Size =", 1, float
        mapInformation["xLL"] = "Lower Left", 0, float
        mapInformation["yLL"] = "Lower Left", 1, float
        mapInformation["xUR"] = "Upper Right", 0, float
        mapInformation["yUR"] = "Upper Right", 1, float
        mapInformation["dataType"] = "Type=", 0, str
        mapInformation["minValue"] = "Min=", 0, float
        mapInformation["maxValue"] = "Max=", 0, float
        mapInformation["noDataValue"] = "NoData Value=", 0, float
        mapAttributes = {}
        # get information with gdalinfo
        command = f'gdalinfo "{inputFileName}"'
        cOut, err = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        ).communicate()

        err = err.decode("utf-8")
        cOut = cOut.decode("utf-8")

        if len(err) > 0 and not err[:7].lower() == "warning":
            sys.exit(
                "Error: no information could be retrieved for the spatial dataset %s"
                % inputFileName
            )

        for mapAttribute, info in mapInformation.items():
            matched = False
            rawList = cOut.split("\n")[:maxLength]
            key = info[0]
            entryNumber = info[1]
            typeInfo = info[2]
            # iterate until the entry is found or the list is empty
            while not matched and len(rawList) > 0:
                # pop the first entry, stripping whitespace (including \r)
                entry = rawList.pop(0).strip()
                if key in entry:
                    matched = True
                    posCnt = entry.find(key)
                    if posCnt >= 0:
                        posCnt += len(key)
                        rawStr = entry[posCnt:].split(",")[entryNumber]
                        rawStr = rawStr.strip("=:() \t")
                        if typeInfo in [int, float]:
                            rawStr = rawStr.split()[0]
                        # process the raw string of the entry
                        rawStr = rawStr.strip("=:() \t")
                        if typeInfo == int:
                            try:
                                mapAttributes[mapAttribute] = int(rawStr)
                            except Exception:
                                sys.exit(
                                    "Error: map attributes could not be processed for %s"
                                    % mapAttribute
                                )
                        elif typeInfo == float:
                            try:
                                mapAttributes[mapAttribute] = float(rawStr)
                            except Exception:
                                sys.exit(
                                    "Error: map attributes could not be processed for %s"
                                    % mapAttribute
                                )
                        else:
                            mapAttributes[mapAttribute] = rawStr
            if mapAttribute in ["xResolution", "yResolution"]:
                mapAttributes[mapAttribute] = abs(mapAttributes[mapAttribute])
            if mapAttribute in mapAttributes.keys():
                setattr(self, mapAttribute, mapAttributes[mapAttribute])
            else:
                setattr(self, mapAttribute, None)


class spatialDataSet:
    """Handles the processing of spatial datasets using GDAL; \
it requires the specification of the bounding box and resolution \
stores data as numpy array in memory under the variable name specified"""

    def __init__(
        self,
        variableName,
        inputFileName,
        typeStr,
        valueScale,
        xLL,
        xUR,
        yLL,
        yUR,
        xResolution,
        yResolution,
        pixels=None,
        lines=None,
        xResampleRatio=None,
        yResampleRatio=None,
        resampleMethod="near",
        outputFileName=None,
        band=None,
        attribute=None,
        warp=False,
        test=False,
        sqlStr="",
        compareStr="=",
        burnValue=None,
    ):

        tempFileRoot = f"temp_{variableName}"
        if isinstance(outputFileName, NoneType):
            outputFileName = f"{tempFileRoot}.map"
        if len(sqlStr) > 0 and not isinstance(burnValue, types.NoneType):
            sqlStr = f"-where \"{attribute}{compareStr}'{sqlStr}'\""
            sqlStr += f" -burn {burnValue:f}"
        else:
            sqlStr = ""
        if not isinstance(valueScale, str):
            try:
                valueScale = str(valueScale)
            except Exception:
                valueScale = "SCALAR"
            valueScale = valueScale.upper()

        # handle special cases such as shape files and bands
        if os.path.splitext(inputFileName)[1] == ".shp":
            command = (
                'gdal_rasterize -a %s %s -ot %s -tr %f %f -te %f %f %f %f  "%s" "%s.tif" -q'
                % (
                    attribute,
                    sqlStr,
                    typeStr,
                    xResolution,
                    yResolution,
                    xLL,
                    yLL,
                    xUR,
                    yUR,
                    inputFileName,
                    tempFileRoot,
                )
            )
            if test:
                print(command)
            cOut, err = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            ).communicate()

            err = err.decode("utf-8")
            cOut = cOut.decode("utf-8")

            if err != "" and not err[:7].lower() == "warning":
                sys.exit(
                    "Error: no information could be retrieved from the spatial dataset %s"
                    % inputFileName
                )
            inputFileName = f"{tempFileRoot}.tif"
        elif band != 0 and not isinstance(band, NoneType):
            command = (
                'gdal_translate -ot %s -of PCRaster -b %d -mo VALUESCALE=VS_%s "%s" "%s_%d.map" -q'
                % (typeStr, band, valueScale, inputFileName, tempFileRoot, band)
            )
            if test:
                print(command)
            cOut, err = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            ).communicate()

            if (
                err != ""
                and b"warning" not in err.lower()
                and not os.path.isfile(f"{tempFileRoot}_{band}.map")
            ):
                sys.exit(
                    "Error: no information could be retrieved from the spatial dataset %s"
                    % inputFileName
                )
            inputFileName = f"{tempFileRoot}_{band}.map"

        if warp:
            # warp the dataset and reset the resolution
            command = (
                'gdalwarp -of GTiff -ot %s -te %f %f %f %f -tr %f %f -r %s "%s" "%s2.tif" -q -overwrite -nomd'
                % (
                    typeStr,
                    xLL,
                    yLL,
                    xUR,
                    yUR,
                    xResolution,
                    yResolution,
                    resampleMethod,
                    inputFileName,
                    tempFileRoot,
                )
            )
            if test:
                print(command)
            cOut, err = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            ).communicate()
            if err != "" and not err[:7].lower() == "warning":
                sys.exit(
                    "Error: no information could be retrieved from the spatial dataset %s"
                    % inputFileName
                )
            inputFileName = f"{tempFileRoot}2.tif"
            xResampleRatio = 1.0
            yResampleRatio = 1.0

        # convert to a PCRaster map of the chosen extent
        if not isinstance(pixels, NoneType) and not isinstance(lines, NoneType):
            command = (
                'gdal_translate -ot %s -of PCRaster -mo VALUESCALE=VS_%s -projwin %f %f %f %f -outsize %s %s "%s" "%s" -q'
                % (
                    typeStr,
                    valueScale,
                    xLL,
                    yUR,
                    xUR,
                    yLL,
                    "%d" % pixels,
                    "%d" % lines,
                    inputFileName,
                    outputFileName,
                )
            )
        elif not isinstance(xResampleRatio, NoneType) and not isinstance(
            xResampleRatio, NoneType
        ):
            command = (
                'gdal_translate -ot %s -of PCRaster -mo VALUESCALE=VS_%s -projwin %f %f %f %f -outsize %s %s "%s" "%s" -q'
                % (
                    typeStr,
                    valueScale,
                    xLL,
                    yUR,
                    xUR,
                    yLL,
                    "%.3f%s" % (100.0 * xResampleRatio, "%"),
                    "%.3f%s" % (100.0 * yResampleRatio, "%"),
                    inputFileName,
                    outputFileName,
                )
            )
        else:
            sys.exit(
                "Error: no information on resample ratio / output extent is specified"
            )
        if test:
            print(command)
        cOut, err = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        ).communicate()

        if len(err) > 0 and b"warning" not in err.lower():
            sys.exit(
                "Error: no information could be retrieved from the spatial dataset %s"
                % inputFileName
            )
        setattr(self, variableName, pcr.readmap(outputFileName))
        for tempFileName in os.listdir(os.getcwd()):
            if os.path.splitext(tempFileName)[1] in [".map", ".tif", ".xml"]:
                if tempFileRoot in tempFileName:
                    try:
                        os.remove(tempFileName)
                    except Exception:
                        pass
