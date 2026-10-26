import datetime
import logging

import numpy as np
import pcraster as pcr

from dynqual.dynqual_main import WaterQuality
from pcrglobwb import waterBodies
from pcrglobwb.common import virtualOS as vos
from pcrglobwb.ncConverter import PCR2netCDF

logger = logging.getLogger(__name__)


class Routing(object):

    def getPseudoState(self):
        result = {}
        return result

    def getState(self):
        result = {}

        # hydrology states
        # (day)
        result["timestepsToAvgDischarge"] = self.timestepsToAvgDischarge
        # channel storage, including lake and reservoir storage (m3)
        result["channelStorage"] = self.channelStorage
        # readily available channel storage that can be abstracted to satisfy water demand (m3)
        result["readAvlChannelStorage"] = self.readAvlChannelStorage
        # long-term average discharge (m3/s)
        result["avgDischargeLong"] = self.avgDischarge
        # ((m3/s)^2)
        result["m2tDischargeLong"] = self.m2tDischarge
        # long-term average baseflow (m3/s)
        result["avgBaseflowLong"] = self.avgBaseflow
        # river bed infiltration from surface water bodies to groundwater (m3/day)
        result["riverbedExchange"] = self.riverbedExchange
        # lake and reservoir storage (m3), per water body id (not per cell)
        result["waterBodyStorage"] = self.waterBodyStorage
        # long-term average lake and reservoir outflow (m3/s), per water body id (not per cell)
        result["avgLakeReservoirOutflowLong"] = self.avgOutflow
        # short-term average lake and reservoir inflow (m3/s), per water body id (not per cell)
        result["avgLakeReservoirInflowShort"] = self.avgInflow
        # short-term average discharge (m3/s)
        result["avgDischargeShort"] = self.avgDischargeShort
        # sub-time step discharge (m3/s), needed for the kinematic wave methods
        result["subDischarge"] = self.subDischarge

        # QUAlloc: long-term water availability
        if self.using_qualloc:
            # discharge (m3/s)
            result["discharge"] = self.discharge
            # total runoff (m/day)
            result["runoff"] = self.runoff

        if self.water_quality is not None:
            result.update(self.water_quality.getState())
        return result

    def __init__(self, iniItems, initialConditions, lddMap):
        object.__init__(self)

        self.lddMap = lddMap

        self.tmpDir = iniItems.tmpDir
        self.inputDir = iniItems.globalOptions["inputDir"]

        # option to activate the water balance check
        self.debugWaterBalance = True
        if iniItems.routingOptions["debugWaterBalance"] == "False":
            self.debugWaterBalance = False

        self.method = iniItems.routingOptions["routingMethod"]

        # option to include lakes and reservoirs
        self.includeWaterBodies = True
        if "includeWaterBodies" in list(iniItems.routingOptions.keys()):
            if (
                iniItems.routingOptions["includeWaterBodies"] == "False"
                or iniItems.routingOptions["includeWaterBodies"] == "None"
            ):
                self.includeWaterBodies = False

        # sectors evaluated
        self.includeSectors = {}
        self.includeSectors["industry"] = False
        self.includeSectors["manufacture"] = False
        if iniItems.waterDemandOptions["includeIndustryWaterDemand"] == "True":
            self.includeSectors["industry"] = True
        elif iniItems.waterDemandOptions["includeManufactureWaterDemand"] == "True":
            self.includeSectors["manufacture"] = True

        self.lddMap = vos.readPCRmapClone(
            iniItems.routingOptions["lddMap"],
            self.tmpDir,
            self.inputDir,
            True,
        )
        self.lddMap = pcr.lddrepair(pcr.ldd(self.lddMap))
        self.lddMap = pcr.lddrepair(self.lddMap)

        # complete ldd before clipping to the landmask (see below); needed to route upstream discharge
        # (from other sub-runs) into this basin, as the cells supplying it lie just outside the landmask
        self.ldd_complete = self.lddMap

        if iniItems.globalOptions["landmask"] != "None":
            self.landmask = vos.readPCRmapClone(
                iniItems.globalOptions["landmask"],
                self.tmpDir,
                self.inputDir,
            )
        else:
            self.landmask = pcr.defined(self.lddMap)
        self.landmask = pcr.ifthen(pcr.defined(self.lddMap), self.landmask)
        self.landmask = pcr.cover(self.landmask, pcr.boolean(0))

        self.lddMap = pcr.lddmask(self.lddMap, self.landmask)

        # cell area (m2)
        self.cellArea = vos.readPCRmapClone(
            iniItems.routingOptions["cellAreaMap"],
            self.tmpDir,
            self.inputDir,
        )

        # model resolution (arc-degree)
        self.cellSizeInArcDeg = vos.getCloneAttributes()["cellsize"]

        # maximum number of time steps for long-term average flows (default: 5 years = 1825 days)
        self.maxTimestepsToAvgDischargeLong = 1825.0

        # maximum number of time steps for short-term average values (default: 1 month = 30 days)
        self.maxTimestepsToAvgDischargeShort = 30.0

        routingParameters = ["gradient", "manningsN"]
        for var in routingParameters:
            input = iniItems.routingOptions[str(var)]
            vars(self)[var] = vos.readPCRmapClone(input, self.tmpDir, self.inputDir)

        # parameters to estimate the channel dimensions (used in getRoutingParamAvgDischarge)
        self.eta = 0.25
        self.nu = 0.40
        self.tau = 8.00
        self.phi = 0.58

        # option to use a minimum channel width (m)
        self.minChannelWidth = pcr.scalar(0.0)
        if "minimumChannelWidth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["minimumChannelWidth"] != "None":
                self.minChannelWidth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["minimumChannelWidth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # option to use a predefined channel width (m)
        self.predefinedChannelWidth = None
        if "constantChannelWidth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["constantChannelWidth"] != "None":
                self.predefinedChannelWidth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["constantChannelWidth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # option to use a predefined channel depth (m)
        self.predefinedChannelDepth = None
        if "constantChannelDepth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["constantChannelDepth"] != "None":
                self.predefinedChannelDepth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["constantChannelDepth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # assumption for broad sheet flow in the kinematic wave methods
        self.beta = 0.6

        # channel length (m), approximated by the cell diagonal
        cellSizeInArcMin = self.cellSizeInArcDeg * 60.0
        verticalSizeInMeter = cellSizeInArcMin * 1852.0
        self.cellLengthFD = (
            (self.cellArea / verticalSizeInMeter) ** (2) + (verticalSizeInMeter) ** (2)
        ) ** (0.5)
        self.channelLength = self.cellLengthFD
        # channel length (m)
        if "channelLength" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["channelLength"] != "None":
                self.channelLength = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["channelLength"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    self.channelLength,
                )

        # dist2celllength (m/arcDegree), needed in the accuTravelTime function
        nrCellsDownstream = pcr.ldddist(self.lddMap, pcr.nominal(self.lddMap) == 5, 1.0)
        distanceDownstream = pcr.ldddist(
            self.lddMap, pcr.nominal(self.lddMap) == 5, self.channelLength
        )
        # (m)
        channelLengthDownstream = (self.channelLength + distanceDownstream) / (
            nrCellsDownstream + 1
        )
        # (m/arcDegree)
        self.dist2celllength = channelLengthDownstream / self.cellSizeInArcDeg

        self.distance_to_pit = 0.5 * self.channelLength + distanceDownstream

        # the channel gradient must be >= minGradient
        minGradient = 0.00005
        self.gradient = pcr.max(minGradient, pcr.cover(self.gradient, minGradient))

        self.WaterBodies = waterBodies.WaterBodies(iniItems, self.landmask)

        # crop evaporation coefficient of surface water bodies
        self.no_zero_crop_water_coefficient = True
        if iniItems.routingOptions["cropCoefficientWaterNC"] == "None":
            self.no_zero_crop_water_coefficient = False
        else:
            self.fileCropKC = vos.getFullPath(
                iniItems.routingOptions["cropCoefficientWaterNC"], self.inputDir
            )

        # Courant number criterion for numerical stability in the kinematic wave methods
        self.courantNumber = 0.50

        # empirical values for the minimum number of sub-time steps (m/s)
        design_flood_speed = 5.00
        design_length_of_sub_time_step = pcr.cellvalue(
            pcr.mapminimum(
                self.courantNumber * self.channelLength / design_flood_speed
            ),
            1,
        )[0]
        self.limit_num_of_sub_time_steps = np.ceil(
            vos.secondsPerDay() / design_length_of_sub_time_step
        )
        # at least 24 sub-time steps (hourly, as in Van Beek et al., 2011)
        self.limit_num_of_sub_time_steps = max(24.0, self.limit_num_of_sub_time_steps)

        # minimum number of sub-time steps from the ini file
        if "maxiumLengthOfSubTimeStep" in list(iniItems.routingOptions.keys()):
            maxiumLengthOfSubTimeStep = float(
                iniItems.routingOptions["maxiumLengthOfSubTimeStep"]
            )
            minimum_number_of_sub_time_step = np.ceil(
                vos.secondsPerDay() / maxiumLengthOfSubTimeStep
            )
            self.limit_num_of_sub_time_steps = max(
                minimum_number_of_sub_time_step, self.limit_num_of_sub_time_steps
            )
        self.limit_num_of_sub_time_steps = int(self.limit_num_of_sub_time_steps)

        # critical water height (m) to select a stable sub-time step length in the kinematic wave methods,
        self.critical_water_height = 0.25
        # as in Van Beek et al. (2011)

        # minimum fracWat used to calculate the water height (-)
        self.min_fracwat_for_water_height = 0.001
        self.max_water_height = 50000000

        # minimum crop coefficient of surface water bodies
        self.minCropWaterKC = 0.00
        if "minCropWaterKC" in list(iniItems.routingOptions.keys()):
            self.minCropWaterKC = float(iniItems.routingOptions["minCropWaterKC"])

        # floodplain options
        self.floodPlain = iniItems.routingOptions["dynamicFloodPlain"] == "True"
        if self.floodPlain:

            logger.info("Flood plain extents can vary during the simulation.")

            # Manning's n of the floodplain areas
            input = iniItems.routingOptions["floodplainManningsN"]
            self.floodplainManN = vos.readPCRmapClone(input, self.tmpDir, self.inputDir)

            # reduction parameter of the smoothing interval and error threshold
            self.reductionKK = 0.5
            if "reductionKK" in list(iniItems.routingOptions.keys()):
                self.reductionKK = float(iniItems.routingOptions["reductionKK"])
            self.criterionKK = 40.0
            if "criterionKK" in list(iniItems.routingOptions.keys()):
                self.criterionKK = float(iniItems.routingOptions["criterionKK"])

            # profile of relative elevation above the floodplain per cell (including smoothing parameters)
            (
                self.nrZLevels,
                self.areaFractions,
                self.relZ,
                self.floodVolume,
                self.kSlope,
                self.mInterval,
            ) = self.getElevationProfile(iniItems)

            # bankfull capacity (m3)
            self.predefinedBankfullCapacity = None
            self.usingFixedBankfullCapacity = False
            if iniItems.routingOptions["bankfullCapacity"] != "None":

                self.usingFixedBankfullCapacity = True
                self.predefinedBankfullCapacity = vos.readPCRmapClone(
                    iniItems.routingOptions["bankfullCapacity"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.predefinedBankfullCapacity = pcr.cover(
                    self.predefinedBankfullCapacity, pcr.spatial(pcr.scalar(0.0))
                )

            else:
                msg = "The bankfull channel storage capacity is NOT defined in the configuration file. "

                if (
                    self.predefinedChannelWidth is None
                    or self.predefinedChannelDepth is None
                ):
                    msg += "The bankfull capacity is estimated from average discharge (5 year long term average)."
                else:
                    msg += "The bankfull capacity is estimated from the given channel depth and channel width."
                    self.usingFixedBankfullCapacity = True
                    self.predefinedBankfullCapacity = self.estimateBankfullCapacity(
                        self.predefinedChannelWidth, self.predefinedChannelDepth
                    )

                logger.info(msg)

        # option to limit the flood depth (to avoid unrealistic flood depths)
        self.maxFloodDepth = None
        if "maxFloodDepth" in list(iniItems.routingOptions.keys()):
            self.maxFloodDepth = vos.readPCRmapClone(
                iniItems.routingOptions["maxFloodDepth"],
                self.tmpDir,
                self.inputDir,
            )

        # input file for upstream discharge (from upstream basins)
        self.upstream_discharge_input_files = None
        if (
            "upstream_discharge_input_files" in list(iniItems.routingOptions.keys())
            and iniItems.routingOptions["upstream_discharge_input_files"] != "None"
        ):
            self.upstream_discharge_input_files = iniItems.routingOptions[
                "upstream_discharge_input_files"
            ].split(",")

        self.WWtPlants = False
        if "WWtPlantsNC" in list(iniItems.routingOptions.keys()):
            self.WWtPlants = True
            # wastewater pathways and removal efficiencies (treatment: tertiary, secondary, primary; collected but untreated; basic sanitation; open defecation; direct)
            self.WWtPlantsNC = vos.getFullPath(
                iniItems.routingOptions["WWtPlantsNC"], self.inputDir
            )

        self.using_qualloc = False
        if (
            "using_qualloc" in iniItems.waterManagementOptions
            and iniItems.waterManagementOptions["using_qualloc"] == "True"
        ):
            self.using_qualloc = True

        self.getICs(iniItems, initialConditions)

        if iniItems.routingOptions.get("quality") == "True":
            raise ValueError(
                "[routingOptions] quality is replaced by [dynqualOptions] using_dynqual;"
                " the DynQual keys moved from [routingOptions] and [meteoOptions] to"
                " [dynqualOptions]."
            )
        self.water_quality = None
        dynqual_options = getattr(iniItems, "dynqualOptions", {})
        if dynqual_options.get("using_dynqual") == "True":
            if self.method != "simplifiedKinematicWave":
                raise ValueError(
                    "DynQual (using_dynqual = True) requires routingMethod = "
                    f"simplifiedKinematicWave, not {self.method}."
                )
            logger.info("Water quality modelling initiated.")
            self.water_quality = WaterQuality(
                iniItems,
                initialConditions,
                landmask=self.landmask,
                lddMap=self.lddMap,
                cellArea=self.cellArea,
                waterBodies=self.WaterBodies,
                includeSectors=self.includeSectors,
                criticalWaterHeight=self.critical_water_height,
                channelStorage=self.channelStorage,
                initialWaterHeight=self.eta * pow(self.avgDischarge, self.nu),
            )

        # old-style reporting (useful for debugging)
        self.initiate_old_style_routing_reporting(iniItems)

    def getICs(self, iniItems, iniConditions=None):

        if iniConditions is None:
            logger.info(
                "Reading initial conditions from pcraster maps listed in the .ini file"
            )
            # read the initial conditions from the PCRaster maps in the ini file (at the start of the model)
            self.timestepsToAvgDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["timestepsToAvgDischargeIni"],
                self.tmpDir,
                self.inputDir,
            )

            self.channelStorage = vos.readPCRmapClone(
                iniItems.routingOptions["channelStorageIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.readAvlChannelStorage = vos.readPCRmapClone(
                iniItems.routingOptions["readAvlChannelStorageIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["avgDischargeLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.m2tDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["m2tDischargeLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgBaseflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgBaseflowLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.riverbedExchange = vos.readPCRmapClone(
                iniItems.routingOptions["riverbedExchangeIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial condition introduced in version 2.0.2: avgDischargeShort
            self.avgDischargeShort = vos.readPCRmapClone(
                iniItems.routingOptions["avgDischargeShortIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial conditions needed for the kinematic wave methods
            self.subDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["subDischargeIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial conditions needed for coupling with QUAlloc
            if self.using_qualloc:
                self.discharge = vos.readPCRmapClone(
                    iniItems.routingOptions["dischargeIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.runoff = vos.readPCRmapClone(
                    iniItems.routingOptions["runoffIni"],
                    self.tmpDir,
                    self.inputDir,
                )

        else:
            logger.info("Reading initial conditions from memory.")
            # read the initial conditions from memory
            self.timestepsToAvgDischarge = iniConditions["routing"][
                "timestepsToAvgDischarge"
            ]

            self.channelStorage = iniConditions["routing"]["channelStorage"]
            self.readAvlChannelStorage = iniConditions["routing"][
                "readAvlChannelStorage"
            ]
            self.avgDischarge = iniConditions["routing"]["avgDischargeLong"]
            self.m2tDischarge = iniConditions["routing"]["m2tDischargeLong"]
            self.avgBaseflow = iniConditions["routing"]["avgBaseflowLong"]
            self.riverbedExchange = iniConditions["routing"]["riverbedExchange"]
            self.avgDischargeShort = iniConditions["routing"]["avgDischargeShort"]

            self.subDischarge = iniConditions["routing"]["subDischarge"]

            # initial conditions needed for coupling with QUAlloc
            if self.using_qualloc:
                self.discharge = iniConditions["routing"]["discharge"]
                self.runoff = iniConditions["routing"]["runoff"]

        # values within the landmask
        self.channelStorage = pcr.ifthen(
            self.landmask, pcr.cover(self.channelStorage, 0.0)
        )
        self.readAvlChannelStorage = pcr.ifthen(
            self.landmask, pcr.cover(self.readAvlChannelStorage, 0.0)
        )
        self.avgDischarge = pcr.ifthen(self.landmask, pcr.cover(self.avgDischarge, 0.0))
        self.m2tDischarge = pcr.ifthen(self.landmask, pcr.cover(self.m2tDischarge, 0.0))
        self.avgDischargeShort = pcr.ifthen(
            self.landmask, pcr.cover(self.avgDischargeShort, 0.0)
        )
        self.avgBaseflow = pcr.ifthen(self.landmask, pcr.cover(self.avgBaseflow, 0.0))
        self.riverbedExchange = pcr.ifthen(
            self.landmask, pcr.cover(self.riverbedExchange, 0.0)
        )
        self.subDischarge = pcr.ifthen(self.landmask, pcr.cover(self.subDischarge, 0.0))

        self.readAvlChannelStorage = pcr.min(
            self.readAvlChannelStorage, self.channelStorage
        )
        self.readAvlChannelStorage = pcr.max(self.readAvlChannelStorage, 0.0)

        # QUAlloc
        if self.using_qualloc:
            self.discharge = pcr.ifthen(self.landmask, pcr.cover(self.discharge, 0.0))
            self.runoff = pcr.ifthen(self.landmask, pcr.cover(self.runoff, 0.0))

        # make sure timestepsToAvgDischarge is the same for the entire map
        try:
            self.timestepsToAvgDischarge = pcr.mapmaximum(self.timestepsToAvgDischarge)
        except Exception:
            # try/except because pcr.mapmaximum cannot handle scalar values
            pass

        # for netCDF reporting, timestepsToAvgDischarge must be spatial and scalar (e.g. for pcr2numpy)
        self.timestepsToAvgDischarge = pcr.spatial(
            pcr.scalar(self.timestepsToAvgDischarge)
        )
        self.timestepsToAvgDischarge = pcr.ifthen(
            self.landmask, self.timestepsToAvgDischarge
        )

        # initial conditions of water bodies: short-term average inflow (m3/s) and long-term average outflow (m3/s)
        if iniConditions is None:
            # read the initial conditions from the PCRaster maps in the ini file (at the start of the model)
            self.avgInflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgLakeReservoirInflowShortIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgOutflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgLakeReservoirOutflowLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            if iniItems.routingOptions["waterBodyStorageIni"] is not None:
                self.waterBodyStorage = vos.readPCRmapClone(
                    iniItems.routingOptions["waterBodyStorageIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.waterBodyStorage = pcr.ifthen(
                    self.landmask, pcr.cover(self.waterBodyStorage, 0.0)
                )
            else:
                self.waterBodyStorage = None
        else:
            # read the initial conditions from memory
            self.avgInflow = iniConditions["routing"]["avgLakeReservoirInflowShort"]
            self.avgOutflow = iniConditions["routing"]["avgLakeReservoirOutflowLong"]
            self.waterBodyStorage = iniConditions["routing"]["waterBodyStorage"]

        self.avgInflow = pcr.ifthen(self.landmask, pcr.cover(self.avgInflow, 0.0))
        self.avgOutflow = pcr.ifthen(self.landmask, pcr.cover(self.avgOutflow, 0.0))
        if self.waterBodyStorage is not None:
            self.waterBodyStorage = pcr.ifthen(
                self.landmask, pcr.cover(self.waterBodyStorage, 0.0)
            )

    def estimateBankfullCapacity(self, width, depth, minWidth=5.0, minDepth=1.0):

        # bankfull capacity (m3)
        bankfullCapacity = (
            pcr.max(minWidth, width) * pcr.max(minDepth, depth) * self.channelLength
        )

        return bankfullCapacity

    def getElevationProfile(self, iniItems):

        # profile of relative elevation above the floodplain (per cell); output: dictionaries kSlope,

        # mInterval, relZ and floodVolume with keys iCnt (index):
        # nrZLevels: number of levels; areaFractions (-): fraction of flooded area; relZ (m): relative
        # elevation above the floodplain; floodVolume (m3): flood volume above the bankfull capacity;
        # kSlope (-): slope used in the interpolation; mInterval (m3): smoothing interval of the interpolation

        logger.info("Get the profile of relative elevation (relZ, unit: m) !!!")

        relativeElevationFileNC = (
            # TODO: define the relative elevation files in a netCDF file
            None
        )
        if relativeElevationFileNC is not None:
            # TODO: use a netCDF file
            pass

        if relativeElevationFileNC is None:

            relZFileName = vos.getFullPath(
                iniItems.routingOptions["relativeElevationFiles"],
                iniItems.globalOptions["inputDir"],
            )

            # fractions of flooded areas (-)
            areaFractions = list(
                map(
                    float,
                    str(iniItems.routingOptions["relativeElevationLevels"]).split(","),
                )
            )
            # number of levels
            nrZLevels = len(areaFractions)
            # TODO: read areaFractions and nrZLevels automatically

        # patch the elevations of sills using the floodplain gradient, with local distances deltaX per
        # increment up to z[N] and the sum over sills; fill all lists (including smoothing interval and slopes)

        relZ = [0.0] * nrZLevels
        for iCnt in range(0, nrZLevels):

            if relativeElevationFileNC is None:
                inputName = relZFileName % (areaFractions[iCnt] * 100)
                relZ[iCnt] = vos.readPCRmapClone(inputName, self.tmpDir, self.inputDir)
            if relativeElevationFileNC is not None:
                # TODO: use a netCDF file
                pass

            # cover the elevation values
            relZ[iCnt] = pcr.ifthen(self.landmask, pcr.cover(relZ[iCnt], 0.0))

            # make sure that relZ[iCnt] >= relZ[iCnt-1] (added by Edwin)
            if iCnt > 0:
                relZ[iCnt] = pcr.max(relZ[iCnt], relZ[iCnt - 1])

        # minimum floodplain slope, defined by the longest sill (first used to retrieve the longest
        # cumulative distance)
        deltaX = [self.cellArea**0.5] * nrZLevels
        deltaX[0] = 0.0
        sumX = deltaX[:]
        minSlope = 0.0
        for iCnt in range(nrZLevels):
            if iCnt < nrZLevels - 1:
                deltaX[iCnt] = (
                    areaFractions[iCnt + 1] ** 0.5 - areaFractions[iCnt] ** 0.5
                ) * deltaX[iCnt]
            else:
                deltaX[iCnt] = (1.0 - areaFractions[iCnt - 1] ** 0.5) * deltaX[iCnt]
            if iCnt > 0:
                sumX[iCnt] = pcr.ifthenelse(
                    relZ[iCnt] == relZ[iCnt - 1], sumX[iCnt - 1] + deltaX[iCnt], 0.0
                )
                minSlope = pcr.ifthenelse(
                    relZ[iCnt] == relZ[iCnt - 1],
                    pcr.max(sumX[iCnt], minSlope),
                    minSlope,
                )
        # the floodplain slope is at most the channel gradient (flow is slower on the floodplain)
        minSlope = pcr.min(self.gradient, 0.5 * pcr.max(deltaX[1], minSlope) ** -1.0)

        # add a small increment to the elevation of each sill (except for lakes; TODO: verify this)
        for iCnt in range(nrZLevels):
            relZ[iCnt] = relZ[iCnt] + sumX[iCnt] * pcr.ifthenelse(
                relZ[nrZLevels - 1] > 0.0, minSlope, 0.0
            )
            # make sure that relZ[iCnt] >= relZ[iCnt-1] (added by Edwin)
            if iCnt > 0:
                relZ[iCnt] = pcr.max(relZ[iCnt], relZ[iCnt - 1])

        # slope and smoothing interval between dy = y(i+1)-y(i) and dx = x(i+1)-x(i), based on volume;
        # volume (m3)
        floodVolume = [0.0] * (nrZLevels)
        # smoothing interval (m3)
        mInterval = [0.0] * (nrZLevels)
        # slope (-)
        kSlope = [0.0] * (nrZLevels)
        for iCnt in range(1, nrZLevels):
            floodVolume[iCnt] = (
                floodVolume[iCnt - 1]
                + 0.5
                * (areaFractions[iCnt] + areaFractions[iCnt - 1])
                * (relZ[iCnt] - relZ[iCnt - 1])
                * self.cellArea
            )
            kSlope[iCnt - 1] = (
                areaFractions[iCnt] - areaFractions[iCnt - 1]
            ) / pcr.max(0.001, floodVolume[iCnt] - floodVolume[iCnt - 1])
        for iCnt in range(1, nrZLevels):
            if iCnt < (nrZLevels - 1):
                mInterval[iCnt] = (
                    0.5
                    * self.reductionKK
                    * pcr.min(
                        floodVolume[iCnt + 1] - floodVolume[iCnt],
                        floodVolume[iCnt] - floodVolume[iCnt - 1],
                    )
                )
            else:
                mInterval[iCnt] = (
                    0.5 * self.reductionKK * (floodVolume[iCnt] - floodVolume[iCnt - 1])
                )

        return nrZLevels, areaFractions, relZ, floodVolume, kSlope, mInterval

    def getRoutingParamAvgDischarge(self, avgDischarge, dist2celllength):
        # routing parameters from the long-term average discharge (m3/s); output: channel dimensions and
        # characteristicDistance (input of accuTravelTime)

        yMean = self.eta * pow(avgDischarge, self.nu)
        wMean = self.tau * pow(avgDischarge, self.phi)

        # option to use a predefined channel width (m)
        if self.predefinedChannelWidth is not None:
            wMean = pcr.cover(self.predefinedChannelWidth, wMean)
        wMean = pcr.max(self.minChannelWidth, wMean)

        # average flow width (m); may be used as channel width estimate (assuming rectangular channels)
        wMean = pcr.max(wMean, 0.01)
        wMean = pcr.cover(wMean, 0.01)
        # average flow depth (m); should not be used as channel depth estimate
        yMean = pcr.max(yMean, 0.01)
        yMean = pcr.cover(yMean, 0.01)

        # characteristicDistance (-), used for accutraveltimeflux (discharge: the total material flowing
        # through the cell, m3/s) and accutraveltimestate (storage: the material deposited in the cell, m3);
        # (m/day)
        characteristicDistance = (
            ((yMean * wMean) / (wMean + 2 * yMean)) ** (2.0 / 3.0)
            * ((self.gradient) ** (0.5))
            / self.manningsN
            * vos.secondsPerDay()
        )

        # (arcDeg/day)
        characteristicDistance = pcr.max(
            (self.cellSizeInArcDeg) * 0.000000001,
            characteristicDistance / dist2celllength,
        )

        # characteristicDistance of each lake/reservoir
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areaaverage(characteristicDistance, self.WaterBodies.waterBodyIds),
        )

        # make sure all outflow is released outside lakes and reservoirs
        outlets = pcr.cover(
            pcr.ifthen(pcr.scalar(self.WaterBodies.waterBodyOut) > 0, pcr.boolean(1)),
            pcr.boolean(0),
        )
        distance_to_outlets = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.ldddist(self.lddMap, outlets, pcr.scalar(1.0)),
        )
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.max(
                distance_to_outlets + pcr.downstreamdist(self.lddMap) * 1.50,
                lakeReservoirCharacteristicDistance,
            ),
        )
        characteristicDistance = pcr.cover(
            lakeReservoirCharacteristicDistance, characteristicDistance
        )
        # (arcDeg/day)
        characteristicDistance = pcr.roundup(characteristicDistance * 100.0) / 100.0

        # minimum characteristicDistance
        characteristicDistance = pcr.cover(
            characteristicDistance, 0.1 * self.cellSizeInArcDeg
        )
        characteristicDistance = pcr.max(
            0.100 * self.cellSizeInArcDeg, characteristicDistance
        )

        return (yMean, wMean, characteristicDistance)

    def getCharacteristicDistance(self, yMean, wMean):

        # Manning's coefficient
        usedManningsN = self.manningsN

        # corrected Manning's coefficient
        if self.floodPlain:

            # wetted perimeter
            flood_only_wetted_perimeter = self.floodDepth * (2.0) + pcr.max(
                0.0,
                self.innundatedFraction * self.cellArea / self.channelLength
                - self.channelWidth,
            )
            channel_only_wetted_perimeter = (
                pcr.min(
                    self.channelDepth,
                    vos.getValDivZero(
                        self.channelStorage, self.channelLength * self.channelWidth, 0.0
                    ),
                )
                * 2.0
                + self.channelWidth
            )
            # total channel wetted perimeter (m)
            channel_wetted_perimeter = (
                channel_only_wetted_perimeter + flood_only_wetted_perimeter
            )
            # minimum channel wetted perimeter: 10 cm
            channel_wetted_perimeter = pcr.max(0.1, channel_wetted_perimeter)

            usedManningsN = (
                (channel_only_wetted_perimeter / channel_wetted_perimeter)
                * self.manningsN ** (1.5)
                + (flood_only_wetted_perimeter / channel_wetted_perimeter)
                * self.floodplainManN ** (1.5)
            ) ** (2.0 / 3.0)

        # characteristicDistance (-), used for accutraveltimeflux (discharge: the total material flowing
        # through the cell, m3/s) and accutraveltimestate (storage: the material deposited in the cell, m3);
        # (m/day)
        characteristicDistance = (
            ((yMean * wMean) / (wMean + 2 * yMean)) ** (2.0 / 3.0)
            * ((self.gradient) ** (0.5))
            / usedManningsN
            * vos.secondsPerDay()
        )

        # (arcDeg/day)
        characteristicDistance = pcr.max(
            (self.cellSizeInArcDeg) * 0.000000001,
            characteristicDistance / self.dist2celllength,
        )

        # characteristicDistance of each lake/reservoir
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areaaverage(characteristicDistance, self.WaterBodies.waterBodyIds),
        )
        # make sure all outflow is released outside lakes and reservoirs
        outlets = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyOut) > 0,
                pcr.spatial(pcr.boolean(1)),
            ),
            pcr.spatial(pcr.boolean(0)),
        )
        distance_to_outlets = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.ldddist(self.lddMap, outlets, pcr.scalar(1.0)),
        )

        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.max(
                distance_to_outlets + pcr.downstreamdist(self.lddMap) * 2.50,
                lakeReservoirCharacteristicDistance,
            ),
        )
        lakeReservoirCharacteristicDistance = pcr.areamaximum(
            lakeReservoirCharacteristicDistance, self.WaterBodies.waterBodyIds
        )
        # TODO: calculate lakeReservoirCharacteristicDistance while obtaining the lake and reservoir parameters

        characteristicDistance = pcr.cover(
            lakeReservoirCharacteristicDistance, characteristicDistance
        )

        # note: in accutraveltime, a characteristicDistance (velocity) of 0 gives zero accutraveltimestate
        # and a very high accutraveltimeflux

        # TODO: consider using the downstreamdist function

        # current solution: roundup to ignore zero and very small values (arcDeg/day)
        characteristicDistance = pcr.roundup(characteristicDistance * 100.0) / 100.0

        # minimum characteristicDistance
        characteristicDistance = pcr.cover(
            characteristicDistance, 0.1 * self.cellSizeInArcDeg
        )
        # TODO: check the minimum distance for the accutraveltime function
        characteristicDistance = pcr.max(
            0.100 * self.cellSizeInArcDeg, characteristicDistance
        )

        return characteristicDistance

    def accuTravelTime(self):

        # accuTravelTime routing

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        # at least 1 m3 of water stays, to minimize numerical errors of the float32 PCRaster implementation
        channelStorageThatWillNotMove += self.channelStorage - pcr.rounddown(
            self.channelStorage
        )
        self.channelStorage = pcr.rounddown(self.channelStorage)

        # channelStorage passed to the routing
        channelStorageForAccuTravelTime = pcr.max(0.0, self.channelStorage)
        # TODO: check why the cover operation is needed
        channelStorageForAccuTravelTime = pcr.cover(
            channelStorageForAccuTravelTime, 0.0
        )

        characteristicDistance = self.getCharacteristicDistance(self.yMean, self.wMean)

        # channel discharge (m3/day)
        self.Q = pcr.accutraveltimeflux(
            self.lddMap,
            channelStorageForAccuTravelTime,
            pcr.max(0.0, characteristicDistance),
        )
        self.Q = pcr.cover(self.Q, 0.0)
        # for very small velocities (characteristicDistanceForAccuTravelTime), discharge can be missing, see
        # http://sourceforge.net/p/pcraster/bugs-and-feature-requests/543/ and
        # http://karssenberg.geo.uu.nl/tt/TravelTimeSpecification.htm; also avoid negative discharge (m3/day)
        self.Q = pcr.max(0.0, self.Q)

        # update channelStorage after routing (m3)
        self.channelStorage = pcr.accutraveltimestate(
            self.lddMap,
            channelStorageForAccuTravelTime,
            pcr.max(0.0, characteristicDistance),
        )

        # return channelStorageThatWillNotMove to channelStorage (m3)
        self.channelStorage += channelStorageThatWillNotMove

        # for non-kinematic wave approaches, subDischarge is Q in m3/s
        self.subDischarge = self.Q / vos.secondsPerDay()
        self.subDischarge = pcr.ifthen(self.landmask, self.subDischarge)

    def estimate_length_of_sub_time_step(self):

        # sub-time step length (s), the shorter the better, estimated from the initial or latest
        # sub-time step discharge (m3/s)
        length_of_sub_time_step = pcr.ifthenelse(
            self.subDischarge > 0.0,
            self.water_height * self.dynamicFracWat * self.cellArea / self.subDischarge,
            vos.secondsPerDay(),
        )

        # number of sub-time steps (Rens van Beek's method)
        critical_condition = (
            (length_of_sub_time_step < vos.secondsPerDay())
            & (self.water_height > self.critical_water_height)
            & (self.lddMap != pcr.ldd(5))
        )
        number_of_sub_time_steps = vos.secondsPerDay() / pcr.cover(
            pcr.areaminimum(
                pcr.ifthen(critical_condition, length_of_sub_time_step), self.landmask
            ),
            vos.secondsPerDay() / self.limit_num_of_sub_time_steps,
        )
        number_of_sub_time_steps = 1.25 * number_of_sub_time_steps + 1
        number_of_sub_time_steps = pcr.roundup(number_of_sub_time_steps)
        # at least one sub-time step
        number_of_loops = max(
            1.0, pcr.cellvalue(pcr.mapmaximum(number_of_sub_time_steps), 1)[1]
        )
        number_of_loops = int(max(self.limit_num_of_sub_time_steps, number_of_loops))

        # actual sub-time step length (s)
        length_of_sub_time_step = vos.secondsPerDay() / number_of_loops

        return (length_of_sub_time_step, number_of_loops)

    def simplifiedKinematicWave(self, meteo, landSurface, groundwater):
        """
        The 'simplifiedKinematicWave':
        1. First, assume that all local fluxes has been added to 'channelStorage'. This is done outside of this function/method.
        2. Then, the 'channelStorage' is routed by using 'pcr.kinematic function' with 'lateral_inflow' = 0.0.
        """

        logger.info("Using the simplifiedKinematicWave method!")

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )

        # channelStorage passed to the routing (m3)
        channelStorageForRouting = pcr.max(0.0, self.channelStorage)

        # water height (m), needed to estimate the sub-time step length and the channel wetted area (for
        # alpha and dischargeInitial)
        self.water_height = pcr.min(
            self.max_water_height,
            channelStorageForRouting
            / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            ),
        )

        # sub-time step length (s)
        length_of_sub_time_step, number_of_loops = (
            self.estimate_length_of_sub_time_step()
        )

        for i_loop in range(number_of_loops):

            # alpha parameter and initial discharge for the kinematic wave
            if self.floodPlain:
                self.dynamicFracWat, self.water_height, alpha, dischargeInitial = (
                    self.kinAlpha(channelStorageForRouting)
                )
                self.dynamicFracWat = pcr.min(
                    pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                )
            else:
                # alpha parameter and initial discharge for the kinematic wave
                alpha, dischargeInitial = (
                    self.calculate_alpha_and_initial_discharge_for_kinematic_wave(
                        channelStorageForRouting,
                        self.water_height,
                        self.innundatedFraction,
                        self.floodDepth,
                    )
                )

            # at the lake/reservoir outlets, use the water body outflow
            waterBodyOutflowInM3PerSec = (
                pcr.cover(
                    pcr.ifthen(
                        self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
                    ),
                    0.0,
                )
                / vos.secondsPerDay()
            )
            waterBodyOutflowInM3PerSec = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                waterBodyOutflowInM3PerSec,
            )
            dischargeInitial = pcr.cover(waterBodyOutflowInM3PerSec, dischargeInitial)

            # discharge (m3/s) from the kinematic wave approximation
            self.subDischarge = pcr.kinematic(
                self.lddMap,
                dischargeInitial,
                0.0,
                alpha,
                self.beta,
                1,
                length_of_sub_time_step,
                self.channelLength,
            )
            self.subDischarge = pcr.cover(self.subDischarge, 0.0)
            self.subDischarge = pcr.max(0.0, pcr.cover(self.subDischarge, 0.0))

            # avoid negative channel storage
            self.subDischarge = (
                pcr.min(
                    self.subDischarge * length_of_sub_time_step,
                    pcr.max(
                        0.0,
                        channelStorageForRouting
                        + pcr.upstream(
                            self.lddMap, self.subDischarge * length_of_sub_time_step
                        ),
                    ),
                )
                / length_of_sub_time_step
            )

            # update channelStorage (m3)
            storage_change_in_volume = (
                pcr.upstream(self.lddMap, self.subDischarge * length_of_sub_time_step)
                - self.subDischarge * length_of_sub_time_step
            )
            channelStorageForRouting += storage_change_in_volume
            # only route non-negative channelStorage (otherwise it stays)
            channelStorageThatWillNotMove += pcr.ifthenelse(
                channelStorageForRouting < 0.0, channelStorageForRouting, 0.0
            )
            channelStorageForRouting = pcr.max(0.000, channelStorageForRouting)

            # update the flood fraction and flood depth (without floodplain)
            if not self.floodPlain:
                self.inundatedFraction, self.floodDepth = (
                    self.returnInundationFractionAndFloodDepth(channelStorageForRouting)
                )

                # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
                # reservoir surface water fraction
                self.dynamicFracWat = pcr.cover(
                    pcr.min(1.0, self.WaterBodies.fracWat), 0.0
                )
                # fraction of channel (including its excess above bankfull capacity)
                self.dynamicFracWat += pcr.max(
                    0.0, 1.0 - self.dynamicFracWat
                ) * pcr.max(self.channelFraction, self.innundatedFraction)

                # dynamicFracWat is at most 1
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask,
                    pcr.min(
                        pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                    ),
                )
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask, pcr.min(1.0, self.dynamicFracWat)
                )
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask, pcr.max(1e-6, self.dynamicFracWat)
                )

                # water height for the next loop, needed to estimate the channel wetted area (for alpha and dischargeInitial)
                self.water_height = channelStorageForRouting / (
                    pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                    * self.cellArea
                )

            # total discharge volume (m3) until this loop
            if i_loop == 0:
                discharge_volume = pcr.scalar(0.0)
            discharge_volume += self.subDischarge * length_of_sub_time_step

            if self.water_quality is not None:
                self.water_quality.route_sub_step(
                    channelStorageForRouting,
                    self.subDischarge,
                    length_of_sub_time_step,
                )

        # channel discharge (m3/day)
        self.Q = discharge_volume

        # update channelStorage after routing
        self.channelStorage = channelStorageForRouting

        # return channelStorageThatWillNotMove to channelStorage
        self.channelStorage += channelStorageThatWillNotMove

        # channel storage cannot be negative
        self.channelStorage = pcr.max(0.0, self.channelStorage)

    def update(self, landSurface, groundwater, currTimeStep, meteo):

        logger.info("routing in progress")

        # water bodies: get the parameters at the start of each year or simulation; this must be called first
        # to define the initial conditions at the start of the simulation
        if currTimeStep.timeStepPCR == 1:
            initial_conditions_for_water_bodies = self.getState()
            # the last argument is for the initial conditions of lakes/reservoirs
            self.WaterBodies.getParameterFiles(
                currTimeStep,
                self.cellArea,
                self.lddMap,
                initial_conditions_for_water_bodies,
            )

        if (currTimeStep.doy == 1) and (currTimeStep.timeStepPCR > 1):
            self.WaterBodies.getParameterFiles(currTimeStep, self.cellArea, self.lddMap)
        if not self.includeWaterBodies:
            # ignore all lakes and reservoirs
            self.WaterBodies.waterBodyIds = pcr.ifthen(self.landmask, pcr.nominal(-1))

        # downstream demand (m3/s) for reservoirs, estimated from the environmental flow discharge; must be
        # called before updating timestepsToAvgDischarge
        self.downstreamDemand = self.estimate_discharge_for_environmental_flow(
            self.channelStorage
        )

        # routing/channel parameters (based on avgDischarge) and water body fraction, needed for the
        # evaporation from water bodies
        self.yMean, self.wMean, self.characteristicDistance = (
            self.getRoutingParamAvgDischarge(self.avgDischarge, self.dist2celllength)
        )

        # channel width (m), depth (m) and fraction
        self.channelWidth = self.wMean
        self.channelDepth = pcr.max(0.0, self.yMean)
        self.channelFraction = pcr.max(
            0.0, pcr.min(1.0, self.channelWidth * self.channelLength / (self.cellArea))
        )

        # water height for the first time step
        if currTimeStep.timeStepPCR == 1:
            if self.floodPlain:
                self.dynamicFracWat, self.water_height = self.returnFloodedFraction(
                    self.channelStorage
                )
                self.dynamicFracWat = pcr.min(
                    pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                )
            else:
                self.dynamicFracWat = pcr.max(
                    self.channelFraction, self.WaterBodies.fracWat
                )
            self.dynamicFracWat = pcr.ifthen(self.landmask, self.dynamicFracWat)

        # channel bankfull capacity (m3)
        if self.floodPlain:
            if self.usingFixedBankfullCapacity:
                self.channelStorageCapacity = self.predefinedBankfullCapacity
            else:
                self.channelStorageCapacity = self.estimateBankfullCapacity(
                    self.channelWidth, self.channelDepth
                )

        # flood inundation fraction (-) and depth (m)
        self.innundatedFraction, self.floodDepth = (
            self.returnInundationFractionAndFloodDepth(self.channelStorage)
        )
        # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
        # reservoir surface water fraction
        self.dynamicFracWat = pcr.cover(pcr.min(1.0, self.WaterBodies.fracWat), 0.0)
        # fraction of channel (including its excess above bankfull capacity)
        self.dynamicFracWat += pcr.max(0.0, 1.0 - self.dynamicFracWat) * pcr.max(
            self.channelFraction, self.innundatedFraction
        )
        # dynamicFracWat is at most 1
        self.dynamicFracWat = pcr.ifthen(
            self.landmask,
            pcr.min(pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0),
        )
        self.dynamicFracWat = pcr.ifthen(
            self.landmask, pcr.min(1.0, self.dynamicFracWat)
        )
        self.dynamicFracWat = pcr.ifthen(
            self.landmask, pcr.max(1e-6, self.dynamicFracWat)
        )

        # routing methods
        if self.method == "accuTravelTime" or self.method == "simplifiedKinematicWave":
            self.simple_update(landSurface, groundwater, currTimeStep, meteo)
        if self.method == "kinematicWave":
            self.kinematic_wave_update(landSurface, groundwater, currTimeStep, meteo)
        # note: this method requires abstraction from fossil groundwater

        # infiltration from surface water bodies (channels, lakes and reservoirs) to groundwater, passed on in
        # the next time step
        self.calculate_exchange_to_groundwater(groundwater, currTimeStep)

        # volume released in pits (losses to the ocean/endorheic basins)
        self.outgoing_volume_at_pits = pcr.ifthen(
            self.landmask, pcr.cover(pcr.ifthen(self.lddMap == pcr.ldd(5), self.Q), 0.0)
        )

        # volume that can be abstracted in the next time step
        self.readAvlChannelStorage = pcr.max(
            0.0, self.estimate_available_volume_for_abstraction(self.channelStorage)
        )

        # water body balance check
        self.waterBodyBalance = self.WaterBodies.waterBodyBalance

        # total runoff from local land surface runoff and local changes in water bodies (m)
        self.totalRunoff = self.local_input_to_surface_water / self.cellArea

        # statistics of long- and short-term flow values
        self.calculate_statistics(groundwater, landSurface)

        # old-style reporting; TODO: remove
        self.old_style_routing_reporting(currTimeStep)

    def calculate_potential_evaporation(
        self, landSurface, currTimeStep, meteo, definedDynamicFracWat=None
    ):

        if not self.no_zero_crop_water_coefficient:
            self.waterKC = 0.0

        # potential evaporation from water bodies; if landSurface.actualET < waterKC *
        # meteo.referencePotET * fracWat, more evaporation is added
        if (
            (currTimeStep.day == 1) or (currTimeStep.timeStepPCR == 1)
        ) and self.no_zero_crop_water_coefficient:
            waterKC = vos.netcdf2PCRobjClone(
                self.fileCropKC,
                "kc",
                currTimeStep.fulldate,
                useDoy="month",
            )
            self.waterKC = pcr.ifthen(self.landmask, pcr.cover(waterKC, 0.0))
            self.waterKC = pcr.max(self.minCropWaterKC, self.waterKC)

        # potential evaporation from water bodies (m/day), reduced by the evaporation calculated in the
        # landSurface module; not over the entire cell area
        waterBodyPotEvapOvesSurfaceWaterArea = pcr.ifthen(
            self.landmask,
            pcr.max(0.0, self.waterKC * meteo.referencePotET - landSurface.actualET),
        )

        # potential evaporation from water bodies over the entire cell area (m/day)
        if definedDynamicFracWat is None:
            dynamicFracWat = self.dynamicFracWat
        waterBodyPotEvap = pcr.max(
            0.0, waterBodyPotEvapOvesSurfaceWaterArea * dynamicFracWat
        )
        return waterBodyPotEvap

    def calculate_evaporation(self, landSurface, groundwater, currTimeStep, meteo):

        # potential evaporation from water bodies over the entire cell area, not only the water bodies (m/day)
        self.waterBodyPotEvap = self.calculate_potential_evaporation(
            landSurface, currTimeStep, meteo
        )

        # evaporation volume from water bodies (m3), not limited to the available channelStorage
        volLocEvapWaterBody = self.waterBodyPotEvap * self.cellArea
        # limited to the available channelStorage
        volLocEvapWaterBody = pcr.min(
            pcr.max(0.0, self.channelStorage), volLocEvapWaterBody
        )

        # update channelStorage (m3) after evaporation from water bodies
        self.channelStorage = self.channelStorage - volLocEvapWaterBody
        self.local_input_to_surface_water -= volLocEvapWaterBody

        # evaporation from water bodies (m)
        self.waterBodyEvaporation = volLocEvapWaterBody / self.cellArea
        self.waterBodyEvaporation = pcr.ifthen(self.landmask, self.waterBodyEvaporation)

        # remaining potential evaporation from water bodies (m)
        self.remainWaterBodyPotEvap = pcr.max(
            0.0, self.waterBodyPotEvap - self.waterBodyEvaporation
        )

    def calculate_exchange_to_groundwater(self, groundwater, currTimeStep):

        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # riverbed infiltration (m3/day), following Inge's principle: only if 0 < baseflow < total groundwater abstraction
        # (fossil and non-fossil); the rate is based on the aquifer saturated conductivity, limited to fracWat
        # and the available channelStorage, and passed to groundwater in the next time step (de Graaf et al.,
        # 2014; Wada et al., 2012; Wada et al., 2010); TODO: improve this concept
        # (m/day)
        riverbedConductivity = groundwater.riverBedConductivity
        # maximum conductivity of 0.1 m/day (Marc Bierkens: resistance of 1 day for a 0.1 m river bed)
        riverbedConductivity = pcr.min(0.1, riverbedConductivity)
        # (m)
        total_groundwater_abstraction = pcr.max(
            0.0,
            groundwater.nonFossilGroundwaterAbs + groundwater.fossilGroundwaterAbstr,
        )
        self.riverbedExchange = pcr.max(
            0.0,
            pcr.min(
                pcr.max(0.0, self.channelStorage),
                pcr.ifthenelse(
                    groundwater.baseflow > 0.0,
                    pcr.ifthenelse(
                        total_groundwater_abstraction > groundwater.baseflow,
                        riverbedConductivity * self.dynamicFracWat * self.cellArea,
                        0.0,
                    ),
                    0.0,
                ),
            ),
        )
        self.riverbedExchange = pcr.cover(self.riverbedExchange, 0.0)
        # avoid flip-flopping
        factor = 0.25
        self.riverbedExchange = pcr.min(
            self.riverbedExchange,
            (1.0 - factor) * pcr.max(0.0, self.channelStorage),
        )
        self.riverbedExchange = pcr.ifthenelse(
            self.channelStorage < 0.0, 0.0, self.riverbedExchange
        )
        self.riverbedExchange = pcr.cover(self.riverbedExchange, 0.0)
        self.riverbedExchange = pcr.ifthen(self.landmask, self.riverbedExchange)

        # update channelStorage (m3) after riverbedExchange (m3)
        self.channelStorage -= self.riverbedExchange
        self.local_input_to_surface_water -= self.riverbedExchange

        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [pcr.scalar(0.0)],
                [self.riverbedExchange / self.cellArea],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage after surface water infiltration",
                True,
                currTimeStep.fulldate,
                threshold=1e-4,
            )

    def return_flows_to_wastewater_treatment_plants(self, currTimeStep, landSurface):

        # wastewater treatment plants: ids
        if currTimeStep.doy == 1:
            # wastewater treatment plant (point) locations
            self.WWt_plantID = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "plant_id",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.WWt_plantID = pcr.nominal(self.WWt_plantID)

            # wastewater treatment plant service zones
            self.WWt_zoneID = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "zone_id",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.WWt_zoneID = pcr.nominal(self.WWt_zoneID)

        # return flow: the sum of the domestic and manufacturing return flows only
        self.nonIrrReturnFlow = (
            landSurface.nonIrrReturnFlowVolumePerSector["domestic"]
            + landSurface.nonIrrReturnFlowVolumePerSector["manufacture"]
        )

        # redirect the return flow to the wastewater treatment plant locations, with removal at the plant (m3/day)
        self.nonIrrReturnFlow = pcr.ifthenelse(
            # cells not in a wastewater treatment zone
            self.WWt_zoneID == 0,
            self.nonIrrReturnFlow,
            pcr.ifthenelse(
                self.WWt_zoneID
                # accumulate water over the treatment zone to the plant location
                == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.nonIrrReturnFlow, 0),
                    self.WWt_zoneID,
                ),
                # after accumulation, set locations without a treatment plant to 0
                0.0,
            ),
        )

        # return flow of the non-irrigation water demand, calculated in landSurface.py (m3)
        self.nonIrrReturnFlow = (
            self.nonIrrReturnFlow
            + landSurface.nonIrrReturnFlowVolumePerSector["livestock"]
            + landSurface.nonIrrReturnFlowVolumePerSector["thermoelectric"]
            + landSurface.nonIrrReturnFlowVolumePerSector["industry"]
        )

    def simple_update(self, landSurface, groundwater, currTimeStep, meteo):

        # update the time steps for the long- and short-term statistics (avgDischarge, avgInflow, avgOutflow, etc.)
        self.timestepsToAvgDischarge += 1.0

        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # total local change (input) to surface water storage (m3); only local processes, no routing;
        # initialized at zero
        self.local_input_to_surface_water = pcr.scalar(0.0)

        # runoff from land surface cells (m/day)
        self.runoff = landSurface.landSurfaceRunoff + groundwater.baseflow

        # update channelStorage (m3) after runoff
        self.channelStorage += self.runoff * self.cellArea
        self.local_input_to_surface_water += self.runoff * self.cellArea

        # upstream discharge (m3/s)
        total_upstream_discharge = pcr.spatial(pcr.scalar(0.0))
        if self.upstream_discharge_input_files is not None:
            for i_ups_file in range(0, len(self.upstream_discharge_input_files)):
                upstream_discharge_input_file = self.upstream_discharge_input_files[
                    i_ups_file
                ]
                self.upstream_discharge = vos.readUpstreamDischarge(
                    upstream_discharge_input_file,
                    "automatic",
                    str(currTimeStep.fulldate),
                    useDoy=None,
                )
                total_upstream_discharge = total_upstream_discharge + pcr.cover(
                    self.upstream_discharge, 0.0
                )
            # add the upstream discharge to the current basin
            total_upstream_discharge = pcr.upstream(
                self.ldd_complete, total_upstream_discharge
            )
        # only values within the landmask
        self.total_upstream_discharge = pcr.ifthen(
            self.landmask, total_upstream_discharge
        )
        # add the upstream discharge to channelStorage (m3)
        self.channelStorage = (
            pcr.cover(self.total_upstream_discharge, 0.0) * 3600.0 * 24.0
            + self.channelStorage
        )

        # update channelStorage (m3) after actSurfaceWaterAbstraction
        self.channelStorage -= landSurface.actSurfaceWaterAbstract * self.cellArea
        self.local_input_to_surface_water -= (
            landSurface.actSurfaceWaterAbstract * self.cellArea
        )

        # channelStorage after surface water abstraction, for reporting (m3)
        self.channelStorageAfterAbstraction = pcr.ifthen(
            self.landmask, self.channelStorage
        )

        # redirect the return flow to the wastewater treatment plant locations (m3)
        if self.WWtPlants:
            self.return_flows_to_wastewater_treatment_plants(currTimeStep, landSurface)
        else:
            self.nonIrrReturnFlow = landSurface.nonIrrReturnFlowVolume

        # add the return flows to channelStorage (m3)
        self.channelStorage += self.nonIrrReturnFlow
        self.local_input_to_surface_water += self.nonIrrReturnFlow

        # evaporation from water bodies (sets self.waterBodyEvaporation, in m)
        self.calculate_evaporation(landSurface, groundwater, currTimeStep, meteo)

        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [self.runoff, self.nonIrrReturnFlow / self.cellArea],
                [landSurface.actSurfaceWaterAbstract, self.waterBodyEvaporation],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage (unit: m) before lake/reservoir outflow",
                True,
                currTimeStep.fulldate,
                threshold=5e-3,
            )

        # lake and reservoir operations
        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # move channelStorage to waterBodyStorage at lake and reservoir cells
        storageAtLakeAndReservoirs = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, self.channelStorage
        )
        storageAtLakeAndReservoirs = pcr.cover(storageAtLakeAndReservoirs, 0.0)

        # only non-negative, rounded-down values
        storageAtLakeAndReservoirs = pcr.max(
            0.00, pcr.rounddown(storageAtLakeAndReservoirs)
        )
        # (m3)
        self.channelStorage -= storageAtLakeAndReservoirs

        # update waterBodyStorage (inflow, storage and outflow)
        self.WaterBodies.update(
            storageAtLakeAndReservoirs,
            self.timestepsToAvgDischarge,
            self.maxTimestepsToAvgDischargeShort,
            self.maxTimestepsToAvgDischargeLong,
            currTimeStep,
            self.avgDischarge,
            vos.secondsPerDay(),
            self.downstreamDemand,
        )

        # waterBodyStorage (m3) after outflow, per water body id (not per cell)
        self.waterBodyStorage = pcr.ifthen(
            self.landmask, self.WaterBodies.waterBodyStorage
        )

        # transfer the outflow from lakes and reservoirs to channelStorage (m3/day)
        waterBodyOutflow = pcr.cover(
            pcr.ifthen(
                self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
            ),
            0.0,
        )

        if self.method == "accuTravelTime":
            # distribute the outflow over the lake/reservoir cells, to avoid waterBodyOutflow skipping cells
            waterBodyOutflow = pcr.areaaverage(
                waterBodyOutflow, self.WaterBodies.waterBodyIds
            )
            waterBodyOutflow = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, waterBodyOutflow
            )
        # (m3/day)
        self.waterBodyOutflow = pcr.cover(waterBodyOutflow, 0.0)

        # update channelStorage (m3) after waterBodyOutflow (m3)
        self.channelStorage += self.waterBodyOutflow
        # note: local_input_to_surface_water does not include waterBodyOutflow

        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [self.waterBodyOutflow / self.cellArea],
                [storageAtLakeAndReservoirs / self.cellArea],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage (unit: m) after lake reservoir/outflow fluxes (errors here are most likely due to pcraster implementation in float_32)",
                True,
                currTimeStep.fulldate,
                threshold=1e-3,
            )

        if self.water_quality is not None:
            self.waterBodyEvaporation = self.water_quality.update_before_routing(
                meteo,
                landSurface,
                groundwater,
                currTimeStep,
                channelStorage=self.channelStorage,
                waterBodyStorage=self.waterBodyStorage,
                runoff=self.runoff,
                dynamicFracWat=self.dynamicFracWat,
                water_height=self.water_height,
                waterBodyEvaporation=self.waterBodyEvaporation,
            )

        # routing: returns the new channelStorage (still without waterBodyStorage) and self.Q, the channel
        # discharge (m3/day)
        if self.method == "accuTravelTime":
            self.accuTravelTime()
        if self.method == "simplifiedKinematicWave":
            self.simplifiedKinematicWave(meteo, landSurface, groundwater)
        # channel discharge of the current time step (m3/s)
        self.discharge = self.Q / vos.secondsPerDay()
        # reported channel discharge cannot be negative
        self.discharge = pcr.max(0.0, self.discharge)
        self.discharge = pcr.ifthen(self.landmask, self.discharge)
        self.disChanWaterBody = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areamaximum(self.discharge, self.WaterBodies.waterBodyIds),
        )
        self.disChanWaterBody = pcr.cover(self.disChanWaterBody, self.discharge)
        self.disChanWaterBody = pcr.ifthen(self.landmask, self.disChanWaterBody)
        # reported channel discharge cannot be negative
        self.disChanWaterBody = pcr.max(0.0, self.disChanWaterBody)

        # return waterBodyStorage to channelStorage
        self.channelStorage = self.WaterBodies.return_water_body_storage_to_channel(
            self.channelStorage
        )

        # reported channel storage cannot be negative
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        if self.water_quality is not None:
            self.water_quality.update_after_routing(
                currTimeStep,
                channelStorage=self.channelStorage,
                dynamicFracWat=self.dynamicFracWat,
                disChanWaterBody=self.disChanWaterBody,
                avgDischarge=self.avgDischarge,
                yMean=self.yMean,
                wMean=self.wMean,
            )

    def calculate_alpha_and_initial_discharge_for_kinematic_wave(
        self, channelStorage, water_height, innundatedFraction, floodDepth
    ):
        # alpha (-), the roughness coefficient of the kinematic wave (see
        # http://pcraster.geo.uu.nl/pcraster/4.0.0/doc/manual/op_kinematic.html), from the wetted area (m2),
        # wetted perimeter (m) and beta (-), assuming a rectangular channel

        # channel wetted area (m2): at least water height x channel width (Edwin), mainly based on
        # channelStorage and channelLength (Rens's approach)
        channel_wetted_area = water_height * self.channelWidth
        channel_wetted_area = pcr.max(
            channel_wetted_area, channelStorage / self.channelLength
        )

        # wetted perimeter
        flood_only_wetted_perimeter = floodDepth * (2.0) + pcr.max(
            0.0,
            innundatedFraction * self.cellArea / self.channelLength - self.channelWidth,
        )
        channel_only_wetted_perimeter = (
            pcr.min(
                self.channelDepth,
                vos.getValDivZero(
                    channelStorage, self.channelLength * self.channelWidth, 0.0
                ),
            )
            * 2.0
            + self.channelWidth
        )
        # total channel wetted perimeter (m)
        channel_wetted_perimeter = (
            channel_only_wetted_perimeter + flood_only_wetted_perimeter
        )
        # minimum channel wetted perimeter: 10 cm
        channel_wetted_perimeter = pcr.max(0.1, channel_wetted_perimeter)

        # corrected Manning's coefficient
        if self.floodPlain:
            usedManningsN = (
                (channel_only_wetted_perimeter / channel_wetted_perimeter)
                * self.manningsN ** (1.5)
                + (flood_only_wetted_perimeter / channel_wetted_perimeter)
                * self.floodplainManN ** (1.5)
            ) ** (2.0 / 3.0)
        else:
            usedManningsN = self.manningsN

        # alpha (-) and initial estimate of the channel discharge (m3/s)
        alpha = (
            usedManningsN
            * channel_wetted_perimeter ** (2.0 / 3.0)
            * self.gradient ** (-0.5)
        ) ** self.beta
        # (m3)
        dischargeInitial = pcr.ifthenelse(
            alpha > 0.0, (channel_wetted_area / alpha) ** (1.0 / self.beta), 0.0
        )

        return (alpha, dischargeInitial)

    def returnInundationFractionAndFloodDepth(self, channelStorage):

        # flood inundation depth above the floodplain (m)
        floodDepth = 0.0

        # channel and flood inundated fraction (-), at least channelFraction
        inundatedFraction = self.channelFraction

        if self.floodPlain:

            # flooded fraction and associated water height for the given channel volume, using a logistic
            # smoother near intersections (K&K, 2007)

            # flood/excess volume above the bankfull capacity (m3)
            excessVolume = pcr.max(0.0, channelStorage - self.channelStorageCapacity)

            # find the match based on the shortest distance to the available intersections or steps
            deltaXMin = self.floodVolume[self.nrZLevels - 1]
            y_i = pcr.scalar(1.0)
            k = [pcr.scalar(0.0)] * 2
            mInt = pcr.scalar(0.0)
            for iCnt in range(self.nrZLevels - 1, 0, -1):
                # find x_i for the current volume and update the match, slope and intercept if applicable
                deltaX = excessVolume - self.floodVolume[iCnt]
                mask = pcr.abs(deltaX) < pcr.abs(deltaXMin)
                deltaXMin = pcr.ifthenelse(mask, deltaX, deltaXMin)
                y_i = pcr.ifthenelse(mask, self.areaFractions[iCnt], y_i)
                k[0] = pcr.ifthenelse(mask, self.kSlope[iCnt - 1], k[0])
                k[1] = pcr.ifthenelse(mask, self.kSlope[iCnt], k[1])
                mInt = pcr.ifthenelse(mask, self.mInterval[iCnt], mInt)

            # scaled deltaX and smoothed function, based on the integrated logistic functions PHI(x) and 1-PHI(x)
            deltaX = deltaXMin
            deltaXScaled = pcr.ifthenelse(
                deltaX < 0.0, pcr.scalar(-1.0), 1.0
            ) * pcr.min(self.criterionKK, pcr.abs(deltaX / pcr.max(1.0, mInt)))
            logInt = self.integralLogisticFunction(deltaXScaled)

            # fractional inundated area
            inundatedFraction = pcr.ifthenelse(
                excessVolume > 0.0,
                pcr.ifthenelse(
                    pcr.abs(deltaXScaled) < self.criterionKK,
                    y_i - k[0] * mInt * logInt[0] + k[1] * mInt * logInt[1],
                    y_i + pcr.ifthenelse(deltaX < 0.0, k[0], k[1]) * deltaX,
                ),
                0.0,
            )
            # at least channelFraction
            inundatedFraction = pcr.max(self.channelFraction, inundatedFraction)
            # at most 1 (-)
            inundatedFraction = pcr.max(0.0, pcr.min(1.0, inundatedFraction))

            # inundation depth above the floodplain (m), zero if excessVolume == 0
            floodDepth = pcr.ifthenelse(
                inundatedFraction > 0.0,
                excessVolume
                / (
                    pcr.max(self.min_fracwat_for_water_height, inundatedFraction)
                    * self.cellArea
                ),
                0.0,
            )

            # maximum flood depth
            if self.maxFloodDepth is not None:
                floodDepth = pcr.max(0.0, pcr.min(self.maxFloodDepth, floodDepth))

        return inundatedFraction, floodDepth

    def kinematic_wave_update(self, landSurface, groundwater, currTimeStep, meteo):

        logger.info("Using the fully kinematic wave method! ")

        # update the time steps for the long- and short-term statistics (avgDischarge, avgInflow, avgOutflow, etc.)
        self.timestepsToAvgDischarge += 1.0

        # total local change (input) to surface water storage (m3); only local processes, no routing;
        # initialized at zero
        self.local_input_to_surface_water = pcr.scalar(0.0)

        # for simplicity, surface water abstraction is done outside the sub-daily time steps;
        # update channelStorage (m3) after actSurfaceWaterAbstraction
        self.channelStorage -= landSurface.actSurfaceWaterAbstract * self.cellArea
        self.local_input_to_surface_water -= (
            landSurface.actSurfaceWaterAbstract * self.cellArea
        )
        # channelStorage after surface water abstraction, for reporting (m3)
        self.channelStorageAfterAbstraction = pcr.ifthen(
            self.landmask, self.channelStorage
        )

        # runoff from land surface cells, over the entire cell area (m/day)
        self.runoff = landSurface.landSurfaceRunoff + groundwater.baseflow

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )

        # channelStorage passed to the routing (m3)
        channelStorageForRouting = pcr.max(0.0, self.channelStorage)

        # water height (m), needed to estimate the sub-time step length
        self.water_height = channelStorageForRouting / (
            pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
            * self.cellArea
        )

        # sub-time step length (s)
        length_of_sub_time_step, number_of_loops = (
            self.estimate_length_of_sub_time_step()
        )

        for i_loop in range(number_of_loops):

            logger.info(f"sub-daily time step {i_loop + 1} from {number_of_loops}")

            if self.debugWaterBalance:
                preStorage = pcr.ifthen(self.landmask, channelStorageForRouting)

            # initialize the accumulated values
            if i_loop == 0:
                # (m3)
                acc_local_input_to_surface_water = pcr.scalar(0.0)
                # (m3)
                acc_water_body_evaporation_volume = pcr.scalar(0.0)
                # (m3)
                acc_discharge_volume = pcr.scalar(0.0)

            # update channelStorageForRouting after runoff and non-irrigation return flow (m3)
            channelStorageForRouting += (
                (self.runoff + landSurface.nonIrrReturnFlow)
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )
            # (m3)
            acc_local_input_to_surface_water += (
                (self.runoff + landSurface.nonIrrReturnFlow)
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )

            # potential evaporation within the sub-time step, over the entire cell area (m)
            water_body_potential_evaporation = (
                self.calculate_potential_evaporation(landSurface, currTimeStep, meteo)
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )

            # accumulate the potential evaporation
            if i_loop == 0:
                self.waterBodyPotEvap = pcr.scalar(0.0)
            self.waterBodyPotEvap += water_body_potential_evaporation

            # update channelStorageForRouting after evaporation
            water_body_evaporation_volume = pcr.min(
                pcr.max(channelStorageForRouting, 0.0),
                water_body_potential_evaporation
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay(),
            )
            channelStorageForRouting -= water_body_evaporation_volume
            acc_local_input_to_surface_water -= water_body_evaporation_volume
            acc_water_body_evaporation_volume += water_body_evaporation_volume

            if self.debugWaterBalance:
                vos.waterBalanceCheck(
                    [
                        self.runoff * length_of_sub_time_step / vos.secondsPerDay(),
                        landSurface.nonIrrReturnFlow
                        * length_of_sub_time_step
                        / vos.secondsPerDay(),
                    ],
                    [water_body_evaporation_volume / self.cellArea],
                    [preStorage / self.cellArea],
                    [channelStorageForRouting / self.cellArea],
                    "channelStorageForRouting (local fluxes)",
                    True,
                    currTimeStep.fulldate,
                    threshold=5e-5,
                )

            # move channelStorageForRouting (m3) to waterBodyStorage at lake and reservoir cells
            storageAtLakeAndReservoirs = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                channelStorageForRouting,
            )
            storageAtLakeAndReservoirs = pcr.cover(storageAtLakeAndReservoirs, 0.0)

            # only non-negative, rounded-down values
            storageAtLakeAndReservoirs = pcr.max(
                0.00, pcr.rounddown(storageAtLakeAndReservoirs)
            )
            channelStorageForRouting = pcr.max(
                0.0, channelStorageForRouting - storageAtLakeAndReservoirs
            )

            # update waterBodyStorage (inflow, storage and outflow)
            self.WaterBodies.update(
                storageAtLakeAndReservoirs,
                self.timestepsToAvgDischarge,
                self.maxTimestepsToAvgDischargeShort,
                self.maxTimestepsToAvgDischargeLong,
                currTimeStep,
                self.avgDischarge,
                length_of_sub_time_step,
                self.downstreamDemand,
            )

            # waterBodyOutflow (m3 per sub-time step) at lake/reservoir outlet cells
            waterBodyOutflow = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
                ),
                0.0,
            )
            waterBodyOutflow = pcr.ifthen(self.landmask, waterBodyOutflow)

            # (m3/s)
            waterBodyOutflowInM3PerSec = waterBodyOutflow / length_of_sub_time_step

            # waterBodyStorage (m3) after outflow, per water body id (not per cell)
            self.waterBodyStorage = pcr.ifthen(
                self.landmask, self.WaterBodies.waterBodyStorage
            )

            # update channelStorage (m3) after waterBodyOutflow (m3); local_input_to_surface_water does not
            # include waterBodyOutflow
            storage_change_in_volume = pcr.upstream(self.lddMap, waterBodyOutflow)
            channelStorageForRouting += storage_change_in_volume

            # water height (m) in channels (excluding lakes and reservoirs), needed to estimate the channel
            # wetted area (for alpha and dischargeInitial)
            self.water_height = channelStorageForRouting / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            )
            # deactivating this line gives negative channelStorage (probably due to too high water heights in lakes and reservoirs)

            # alpha parameter and initial discharge estimate for the kinematic wave
            alpha, dischargeInitial = (
                self.calculate_alpha_and_initial_discharge_for_kinematic_wave(
                    channelStorageForRouting,
                    self.water_height,
                    self.innundatedFraction,
                    self.floodDepth,
                )
            )

            # at lake and reservoir outlet cells, the initial discharge is waterBodyOutflowInM3PerSec
            dischargeInitial = pcr.cover(
                pcr.ifthen(self.WaterBodies.waterBodyOut, waterBodyOutflowInM3PerSec),
                dischargeInitial,
            )

            # also for cells with zero channelStorageForRouting
            dischargeInitial = pcr.ifthenelse(
                channelStorageForRouting > 0.0, dischargeInitial, 0.0
            )

            # (m3/s)
            dischargeInitial = pcr.cover(dischargeInitial, 0.0)
            dischargeInitial = pcr.ifthen(self.landmask, dischargeInitial)

            # discharge (m3/s) from the kinematic wave approximation
            self.subDischarge = pcr.kinematic(
                self.lddMap,
                dischargeInitial,
                0.0,
                alpha,
                self.beta,
                1,
                length_of_sub_time_step,
                self.channelLength,
            )
            self.subDischarge = pcr.max(0.0, pcr.cover(self.subDischarge, 0.0))

            # zero discharge at lake and reservoir cells
            self.subDischarge = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, pcr.scalar(0.0)
                ),
                self.subDischarge,
            )

            # avoid negative channel storage
            self.subDischarge = (
                pcr.min(
                    self.subDischarge * length_of_sub_time_step,
                    pcr.max(
                        0.0,
                        channelStorageForRouting
                        + pcr.upstream(
                            self.lddMap, self.subDischarge * length_of_sub_time_step
                        ),
                    ),
                )
                / length_of_sub_time_step
            )

            # update channelStorage (m3) after lateral flows in channels
            storage_change_in_volume = (
                pcr.upstream(self.lddMap, self.subDischarge * length_of_sub_time_step)
                - self.subDischarge * length_of_sub_time_step
            )
            channelStorageForRouting += storage_change_in_volume

            # return waterBodyStorage to channelStorage
            channelStorageForRouting = (
                self.WaterBodies.return_water_body_storage_to_channel(
                    channelStorageForRouting
                )
            )

            # add waterBodyOutflowInM3PerSec to subDischarge
            self.subDischarge += waterBodyOutflowInM3PerSec
            self.subDischarge = pcr.ifthen(self.landmask, self.subDischarge)

            # total discharge volume (m3) until this loop
            acc_discharge_volume += self.subDischarge * length_of_sub_time_step

            # update the flood fraction and flood depth
            self.inundatedFraction, self.floodDepth = (
                self.returnInundationFractionAndFloodDepth(channelStorageForRouting)
            )

            # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
            # reservoir surface water fraction
            self.dynamicFracWat = pcr.cover(pcr.min(1.0, self.WaterBodies.fracWat), 0.0)

            # fraction of channel (including its excess above bankfull capacity)
            self.dynamicFracWat += pcr.max(0.0, 1.0 - self.dynamicFracWat) * pcr.max(
                self.channelFraction, self.innundatedFraction
            )

            # dynamicFracWat is at most 1
            self.dynamicFracWat = pcr.ifthen(
                self.landmask,
                pcr.min(pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0),
            )
            self.dynamicFracWat = pcr.ifthen(
                self.landmask, pcr.min(1.0, self.dynamicFracWat)
            )

            # for the next loop, only route non-negative channelStorage
            channelStorageThatWillNotMove += pcr.ifthenelse(
                channelStorageForRouting < 0.0, channelStorageForRouting, 0.0
            )
            channelStorageForRouting = pcr.max(0.000, channelStorageForRouting)

            # water height, including lakes and reservoirs
            self.water_height = pcr.max(0.0, channelStorageForRouting) / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            )

        # evaporation (m/day)
        self.waterBodyEvaporation = acc_water_body_evaporation_volume / self.cellArea

        # local input to surface water (m3)
        self.local_input_to_surface_water += acc_local_input_to_surface_water

        # channel discharge (m3/day)
        self.Q = acc_discharge_volume

        # update channelStorage after routing
        self.channelStorage = channelStorageForRouting

        # return channelStorageThatWillNotMove to channelStorage
        self.channelStorage += channelStorageThatWillNotMove

        # reported channel storage cannot be negative
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        # channel discharge of the current time step (m3/s)
        self.discharge = self.Q / vos.secondsPerDay()
        # reported channel discharge cannot be negative
        self.discharge = pcr.max(0.0, self.discharge)
        self.discharge = pcr.ifthen(self.landmask, self.discharge)

        self.disChanWaterBody = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areamaximum(self.discharge, self.WaterBodies.waterBodyIds),
        )
        self.disChanWaterBody = pcr.cover(self.disChanWaterBody, self.discharge)
        self.disChanWaterBody = pcr.ifthen(self.landmask, self.disChanWaterBody)

        # reported channel discharge cannot be negative
        self.disChanWaterBody = pcr.max(0.0, self.disChanWaterBody)

    def calculate_statistics(self, groundwater, landSurface):

        # short-term average inflow (m3/s) and long-term average outflow (m3/s) of lakes and reservoirs
        self.avgInflow = pcr.ifthen(
            self.landmask, pcr.cover(self.WaterBodies.avgInflow, 0.0)
        )
        self.avgOutflow = pcr.ifthen(
            self.landmask, pcr.cover(self.WaterBodies.avgOutflow, 0.0)
        )

        # short- and long-term average discharge (m3/s), using the online algorithm of
        # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        # long-term average discharge
        dishargeUsed = pcr.max(0.0, self.discharge)
        dishargeUsed = pcr.max(dishargeUsed, self.disChanWaterBody)

        deltaAnoDischarge = dishargeUsed - self.avgDischarge
        self.avgDischarge = self.avgDischarge + deltaAnoDischarge / pcr.min(
            self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge
        )
        self.avgDischarge = pcr.max(0.0, self.avgDischarge)
        self.m2tDischarge = self.m2tDischarge + pcr.abs(
            deltaAnoDischarge * (dishargeUsed - self.avgDischarge)
        )

        # short-term average discharge
        deltaAnoDischargeShort = dishargeUsed - self.avgDischargeShort
        self.avgDischargeShort = (
            self.avgDischargeShort
            + deltaAnoDischargeShort
            / pcr.min(
                self.maxTimestepsToAvgDischargeShort, self.timestepsToAvgDischarge
            )
        )
        self.avgDischargeShort = pcr.max(0.0, self.avgDischargeShort)

        # long-term average baseflow (m3/s), used as proxy to partition groundwater and surface water abstractions
        baseflowM3PerSec = groundwater.baseflow * self.cellArea / vos.secondsPerDay()

        deltaAnoBaseflow = baseflowM3PerSec - self.avgBaseflow
        self.avgBaseflow = self.avgBaseflow + deltaAnoBaseflow / pcr.min(
            self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge
        )
        self.avgBaseflow = pcr.max(0.0, self.avgBaseflow)

        if self.water_quality is not None:
            self.water_quality.update_statistics(
                landSurface, self.timestepsToAvgDischarge
            )

    def estimate_discharge_for_environmental_flow(self, channelStorage):

        # statistical assumption: z-score of the 90th percentile
        z_score = 1.2816

        # long-term variance and standard deviation of discharge, using the online algorithm of
        # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        varDischarge = self.m2tDischarge / pcr.max(
            1.0,
            pcr.min(self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge)
            - 1.0,
        )
        stdDischarge = pcr.max(varDischarge**0.5, 0.0)

        # minimum discharge for environmental flow (m3/s)
        minDischargeForEnvironmentalFlow = pcr.max(
            0.0, self.avgDischarge - z_score * stdDischarge
        )
        # avoid flip-flopping
        factor = 0.10
        # (m3/s)
        minDischargeForEnvironmentalFlow = pcr.max(
            factor * self.avgDischarge, minDischargeForEnvironmentalFlow
        )
        minDischargeForEnvironmentalFlow = pcr.max(
            0.0, minDischargeForEnvironmentalFlow
        )

        return minDischargeForEnvironmentalFlow

    def estimate_available_volume_for_abstraction(
        self, channelStorage, length_of_time_step=vos.secondsPerDay()
    ):
        # input: channelStorage (m3)

        # minimum discharge for environmental flow (m3/s)
        minDischargeForEnvironmentalFlow = (
            self.estimate_discharge_for_environmental_flow(channelStorage)
        )

        # channelStorage available for surface water abstraction
        readAvlChannelStorage = pcr.max(0.0, channelStorage)

        # reduce readAvlChannelStorage if the average discharge < minDischargeForEnvironmentalFlow
        readAvlChannelStorage *= pcr.min(
            1.0,
            vos.getValDivZero(
                pcr.max(0.0, pcr.min(self.avgDischargeShort, self.avgDischarge)),
                minDischargeForEnvironmentalFlow,
                vos.smallNumber,
            ),
        )

        # maintain the environmental flow if the average discharge > minDischargeForEnvironmentalFlow; TODO: check why this is needed
        readAvlChannelStorage = pcr.ifthenelse(
            self.avgDischargeShort < minDischargeForEnvironmentalFlow,
            readAvlChannelStorage,
            pcr.max(
                readAvlChannelStorage,
                pcr.max(0.0, self.avgDischargeShort - minDischargeForEnvironmentalFlow)
                * length_of_time_step,
            ),
        )

        # maximum fraction of water that can be abstracted from the channel (to avoid flip-flopping)
        maximum_percentage = 0.90
        readAvlChannelStorage = pcr.min(
            readAvlChannelStorage, maximum_percentage * channelStorage
        )
        readAvlChannelStorage = pcr.max(0.0, readAvlChannelStorage)

        # ignore small volumes (less than 0.1 m3)
        readAvlChannelStorage = pcr.rounddown(readAvlChannelStorage * 10.0) / 10.0
        readAvlChannelStorage = pcr.ifthen(self.landmask, readAvlChannelStorage)

        # (m3)
        return readAvlChannelStorage

    def initiate_old_style_routing_reporting(self, iniItems):

        self.report = True
        try:
            self.outDailyTotNC = iniItems.routingOptions["outDailyTotNC"].split(",")
            self.outMonthTotNC = iniItems.routingOptions["outMonthTotNC"].split(",")
            self.outMonthAvgNC = iniItems.routingOptions["outMonthAvgNC"].split(",")
            self.outMonthEndNC = iniItems.routingOptions["outMonthEndNC"].split(",")
            self.outAnnuaTotNC = iniItems.routingOptions["outAnnuaTotNC"].split(",")
            self.outAnnuaAvgNC = iniItems.routingOptions["outAnnuaAvgNC"].split(",")
            self.outAnnuaEndNC = iniItems.routingOptions["outAnnuaEndNC"].split(",")
        except Exception:
            self.report = False
        if self.report:
            # daily netCDF output
            self.outNCDir = iniItems.outNCDir
            self.netcdfObj = PCR2netCDF(iniItems)
            if self.outDailyTotNC[0] != "None":
                for var in self.outDailyTotNC:
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_dailyTot.nc",
                        var,
                        "undefined",
                    )
            # monthly netCDF output: totals
            if self.outMonthTotNC[0] != "None":
                for var in self.outMonthTotNC:
                    # accumulator
                    vars(self)[f"{var}MonthTot"] = None
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_monthTot.nc",
                        var,
                        "undefined",
                    )
            # averages
            if self.outMonthAvgNC[0] != "None":
                for var in self.outMonthAvgNC:
                    # accumulator
                    vars(self)[f"{var}MonthTot"] = None
                    vars(self)[f"{var}MonthAvg"] = None
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_monthAvg.nc",
                        var,
                        "undefined",
                    )
            # end of month
            if self.outMonthEndNC[0] != "None":
                for var in self.outMonthEndNC:
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_monthEnd.nc",
                        var,
                        "undefined",
                    )
            # yearly netCDF output: totals
            if self.outAnnuaTotNC[0] != "None":
                for var in self.outAnnuaTotNC:
                    # accumulator
                    vars(self)[f"{var}AnnuaTot"] = None
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_annuaTot.nc",
                        var,
                        "undefined",
                    )
            # averages
            if self.outAnnuaAvgNC[0] != "None":
                for var in self.outAnnuaAvgNC:
                    vars(self)[f"{var}AnnuaAvg"] = None
                    # accumulator
                    vars(self)[f"{var}AnnuaTot"] = None
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_annuaAvg.nc",
                        var,
                        "undefined",
                    )
            # end of year
            if self.outAnnuaEndNC[0] != "None":
                for var in self.outAnnuaEndNC:
                    self.netcdfObj.createNetCDF(
                        f"{self.outNCDir}/{var}_annuaEnd.nc",
                        var,
                        "undefined",
                    )

    def old_style_routing_reporting(self, currTimeStep):

        if self.report:
            timeStamp = datetime.datetime(
                currTimeStep.year, currTimeStep.month, currTimeStep.day, 0
            )
            # daily netCDF output
            timestepPCR = currTimeStep.timeStepPCR
            if self.outDailyTotNC[0] != "None":
                for var in self.outDailyTotNC:
                    self.netcdfObj.data2NetCDF(
                        f"{self.outNCDir}/{var}_dailyTot.nc",
                        var,
                        pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                        timeStamp,
                        timestepPCR - 1,
                    )

            # monthly netCDF output: totals
            if self.outMonthTotNC[0] != "None":
                for var in self.outMonthTotNC:

                    # initialize at the start of the simulation or reset at the start of the month
                    if currTimeStep.timeStepPCR == 1 or currTimeStep.day == 1:
                        vars(self)[f"{var}MonthTot"] = pcr.scalar(0.0)

                    vars(self)[f"{var}MonthTot"] += vars(self)[var]

                    if currTimeStep.endMonth:
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_monthTot.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(f"{var}MonthTot"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )
            # averages
            if self.outMonthAvgNC[0] != "None":
                for var in self.outMonthAvgNC:
                    # only if no accumulator is defined
                    if var not in self.outMonthTotNC:

                        # initialize at the start of the simulation or reset at the start of the month
                        if currTimeStep.timeStepPCR == 1 or currTimeStep.day == 1:
                            vars(self)[f"{var}MonthTot"] = pcr.scalar(0.0)
                        vars(self)[f"{var}MonthTot"] += vars(self)[var]

                    if currTimeStep.endMonth:
                        vars(self)[f"{var}MonthAvg"] = (
                            vars(self)[f"{var}MonthTot"] / currTimeStep.day
                        )
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_monthAvg.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(f"{var}MonthAvg"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )
            # end of month
            if self.outMonthEndNC[0] != "None":
                for var in self.outMonthEndNC:
                    if currTimeStep.endMonth:
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_monthEnd.nc",
                            var,
                            pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )

            # yearly netCDF output: totals
            if self.outAnnuaTotNC[0] != "None":
                for var in self.outAnnuaTotNC:

                    # initialize at the start of the simulation or reset at the start of the year
                    if currTimeStep.timeStepPCR == 1 or currTimeStep.doy == 1:
                        vars(self)[f"{var}AnnuaTot"] = pcr.scalar(0.0)

                    vars(self)[f"{var}AnnuaTot"] += vars(self)[var]

                    if currTimeStep.endYear:
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_annuaTot.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(f"{var}AnnuaTot"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )
            # averages
            if self.outAnnuaAvgNC[0] != "None":
                for var in self.outAnnuaAvgNC:
                    # only if no accumulator is defined
                    if var not in self.outAnnuaTotNC:
                        # initialize at the start of the simulation or reset at the start of the year
                        if currTimeStep.timeStepPCR == 1 or currTimeStep.doy == 1:
                            vars(self)[f"{var}AnnuaTot"] = pcr.scalar(0.0)
                        vars(self)[f"{var}AnnuaTot"] += vars(self)[var]
                    if currTimeStep.endYear:
                        vars(self)[f"{var}AnnuaAvg"] = (
                            vars(self)[f"{var}AnnuaTot"] / currTimeStep.doy
                        )
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_annuaAvg.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(f"{var}AnnuaAvg"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )
            # end of year
            if self.outAnnuaEndNC[0] != "None":
                for var in self.outAnnuaEndNC:
                    if currTimeStep.endYear:
                        self.netcdfObj.data2NetCDF(
                            f"{self.outNCDir}/{var}_annuaEnd.nc",
                            var,
                            pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )

    def returnFloodedFraction(self, channelStorage):
        # flooded fraction and associated water height for the given flood volume, using a logistic smoother
        # near intersections (K&K, 2007); find the match based on the shortest distance to the available
        # intersections or steps
        deltaXMin = self.floodVolume[self.nrZLevels - 1]
        y_i = pcr.scalar(1.0)
        k = [pcr.scalar(0.0)] * 2
        mInt = pcr.scalar(0.0)
        for iCnt in range(self.nrZLevels - 1, 0, -1):
            # find x_i for the current volume and update the match, slope and intercept if applicable
            deltaX = channelStorage - self.floodVolume[iCnt]
            mask = pcr.abs(deltaX) < pcr.abs(deltaXMin)
            deltaXMin = pcr.ifthenelse(mask, deltaX, deltaXMin)
            y_i = pcr.ifthenelse(mask, self.areaFractions[iCnt], y_i)
            k[0] = pcr.ifthenelse(mask, self.kSlope[iCnt - 1], k[0])
            k[1] = pcr.ifthenelse(mask, self.kSlope[iCnt], k[1])
            mInt = pcr.ifthenelse(mask, self.mInterval[iCnt], mInt)

        # scaled deltaX and smoothed function, based on the integrated logistic functions PHI(x) and 1-PHI(x)
        deltaX = deltaXMin
        deltaXScaled = pcr.ifthenelse(deltaX < 0.0, pcr.scalar(-1.0), 1.0) * pcr.min(
            self.criterionKK, pcr.abs(deltaX / pcr.max(1.0, mInt))
        )
        logInt = self.integralLogisticFunction(deltaXScaled)

        # fractional flooded area and flood depth
        floodedFraction = pcr.ifthenelse(
            channelStorage > 0.0,
            pcr.ifthenelse(
                pcr.abs(deltaXScaled) < self.criterionKK,
                y_i - k[0] * mInt * logInt[0] + k[1] * mInt * logInt[1],
                y_i + pcr.ifthenelse(deltaX < 0.0, k[0], k[1]) * deltaX,
            ),
            0.0,
        )
        floodedFraction = pcr.max(0.0, pcr.min(1.0, floodedFraction))
        floodDepth = pcr.ifthenelse(
            floodedFraction > 0.0,
            channelStorage / (floodedFraction * self.cellArea),
            0.0,
        )
        floodDepth = pcr.min(self.max_water_height, floodDepth)

        return floodedFraction, floodDepth

    def integralLogisticFunction(self, x):
        # tuple with the integrals of the logistic functions of x and -x
        logInt = pcr.ln(pcr.exp(-x) + 1)
        return logInt, x + logInt

    def kinAlpha(self, channelStorage):
        mask = pcr.boolean(1)

        # kinAlphaDynamic logic
        floodVol = pcr.max(0, channelStorage - self.channelStorageCapacity)
        floodedFraction, floodDepth = self.returnFloodedFraction(floodVol)

        self.channelFraction = pcr.max(
            0.0, pcr.min(1.0, self.wMean * self.cellLengthFD / self.cellArea)
        )
        floodFrac_dyn = floodedFraction + self.channelFraction

        wetA_dyn = channelStorage / self.channelLength
        wetPFld = (
            pcr.max(
                0.0, floodFrac_dyn * self.cellArea / self.channelLength - self.wMean
            )
            + 2.0 * floodDepth
        )
        wetPCh = self.wMean + 2.0 * pcr.min(
            self.channelDepth, channelStorage / (self.channelLength * self.wMean)
        )
        wetP_dyn = wetPFld + wetPCh
        manQ = (
            wetPCh / wetP_dyn * self.manningsN**1.5
            + wetPFld / wetP_dyn * self.floodplainManN**1.5
        ) ** (2.0 / 3.0)
        dynamicAlphaQ = (
            manQ * wetP_dyn ** (2.0 / 3.0) * self.gradient**-0.5
        ) ** self.beta
        dynamicDischargeInitial = pcr.ifthenelse(
            dynamicAlphaQ > 0.0, (wetA_dyn / dynamicAlphaQ) ** (1 / self.beta), 0.0
        )

        # kinAlphaStatic logic
        if self.water_quality is not None:
            manIce = pcr.max(
                self.manningsN,
                0.0493
                * pcr.max(
                    0.01, self.channelStorage / (self.dynamicFracWat * self.cellArea)
                )
                ** (-0.23)
                * self.water_quality.iceThickness**0.57,
            )
            manningsWithIce = (0.5 * (self.manningsN**1.5 + manIce**1.5)) ** (2.0 / 3.0)
            wetA_sta = self.channelStorage / self.channelLength
            wetP_sta = 2.0 * wetA_sta / self.wMean + self.wMean
            staticAlphaQ = (
                manningsWithIce * wetP_sta ** (2.0 / 3.0) * self.gradient**-0.5
            ) ** self.beta
        else:
            wetA_sta = channelStorage / self.channelLength
            wetP_sta = 2.0 * wetA_sta / self.wMean + self.wMean
            staticAlphaQ = (
                self.manningsN * wetP_sta ** (2.0 / 3.0) * self.gradient**-0.5
            ) ** self.beta

        staticDischargeInitial = pcr.ifthenelse(
            staticAlphaQ > 0.0, (wetA_sta / staticAlphaQ) ** (1 / self.beta), 0.0
        )

        # combine using the mask
        floodedFraction = pcr.ifthenelse(mask, floodFrac_dyn, 0.0)
        floodDepth = pcr.ifthenelse(mask, floodDepth, 0.0)
        alphaQ = pcr.ifthenelse(mask, dynamicAlphaQ, staticAlphaQ)
        dischargeInitial = pcr.ifthenelse(
            mask, dynamicDischargeInitial, staticDischargeInitial
        )

        return floodedFraction, floodDepth, alphaQ, dischargeInitial
