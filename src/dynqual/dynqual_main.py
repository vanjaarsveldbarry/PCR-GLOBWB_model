import logging
from copy import deepcopy

import pcraster as pcr

from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)


class WaterQuality(object):
    """DynQual: water temperature, salinity (TDS), organic (BOD) and pathogen (FC) pollution.

    Stepped by Routing, which passes the hydrological state into each update_* method.
    Requires routingMethod = simplifiedKinematicWave: pollutants are only transported in
    route_sub_step, which only that method calls.
    """

    def __init__(
        self,
        iniItems,
        initialConditions,
        landmask,
        lddMap,
        cellArea,
        waterBodies,
        includeSectors,
        criticalWaterHeight,
        channelStorage,
        initialWaterHeight,
    ):
        options = iniItems.dynqualOptions

        self.inputDir = iniItems.globalOptions["inputDir"]
        self.tmpDir = iniItems.tmpDir
        self.landmask = landmask
        self.lddMap = lddMap
        self.cellArea = cellArea
        self.WaterBodies = waterBodies
        self.includeSectors = includeSectors
        self.critical_water_height = criticalWaterHeight

        self.using_qualloc = (
            iniItems.waterManagementOptions.get("using_qualloc") == "True"
        )
        self.WWtPlantsNC = vos.getFullPath(
            iniItems.routingOptions["WWtPlantsNC"], self.inputDir
        )

        # discharge threshold for estimating concentrations (default: 0.1 m3/s)
        self.WQ_discharge_threshold = 0.1
        if "WQ_discharge_threshold" in options:
            self.WQ_discharge_threshold = float(options["WQ_discharge_threshold"])

        # water temperature parameters: temperature threshold for snowmelt (K)
        self.iceThresTemp = pcr.scalar(273.15)
        # density of water (kg/m3)
        self.densityWater = pcr.scalar(1000.0)
        # latent heat of vaporization (J/kg)
        self.latentHeatVapor = pcr.scalar(2.5e6)
        # latent heat of fusion (J/kg)
        self.latentHeatFusion = pcr.scalar(3.34e5)
        # specific heat of water (J/kg/degC)
        self.specificHeatWater = pcr.scalar(4190.0)
        # heat transfer coefficient of water (W/m2/degC)
        self.heatTransferWater = pcr.scalar(20.0)
        # heat transfer coefficient of ice (W/m2/degC)
        self.heatTransferIce = pcr.scalar(8.0)
        # albedo of water (-)
        self.albedoWater = pcr.scalar(0.15)
        # albedo of snow and ice (-)
        self.albedoSnow = pcr.scalar(0.50)
        # energy balance proxy for the groundwater temperature: mean annual temperature and the temperature reduction for falling rain
        self.deltaTPrec = pcr.scalar(1.5)

        self.radCon = 0.25
        self.radSlope = 0.50
        # Stefan-Boltzmann constant (W/m2/K4)
        self.stefanBoltzman = 5.67e-8
        # maximum river temperature: 322.15 K (50 degC)
        self.maxThresTemp = pcr.scalar(322.15)

        self.maxIceThickness = 3.0
        self.deltaIceThickness = 0.0

        # convert cloud cover to sunshine hours (Doornkamp & Pruitt)
        self.sunFracTBL = vos.getFullPath(options["sunhoursTable"], self.inputDir)

        # paths to additional meteorological variables
        self.cloudFileNC = vos.getFullPath(options["cloudcoverNC"], self.inputDir)
        self.radFileNC = vos.getFullPath(options["radiationNC"], self.inputDir)
        self.vapFileNC = vos.getFullPath(options["vaporNC"], self.inputDir)
        self.annualTFileNC = vos.getFullPath(options["annualAvgTNC"], self.inputDir)

        # salinity: background TDS concentration (mg/L)
        self.backgroundSalinityNC = vos.getFullPath(
            options["backgroundSalinity"], self.inputDir
        )
        # (mg/L)
        self.backgroundSalinity = vos.netcdf2PCRobjCloneWithoutTime(
            self.backgroundSalinityNC, "bgTDS"
        )

        # organic pollution: first-order degradation coefficient at 20 degC (van Vliet et al., 2021)
        self.k_BOD = pcr.scalar(0.35)
        # temperature correction (van Vliet et al., 2021; Wen et al., 2017)
        self.watertempcorrection_BOD = pcr.scalar(1.047)

        # dissolved oxygen: elevation data
        self.elevation_path = vos.getFullPath(
            iniItems.landSurfaceOptions["topographyNC"], self.inputDir
        )
        self.elevation = vos.netcdf2PCRobjCloneWithoutTime(
            self.elevation_path,
            "dem_average",
            True,
            None,
            self.inputDir,
        )

        # fecal coliforms: temperature-dependent decay (day-1; Reder et al., 2015)

        self.darkinactivation_FC = pcr.scalar(0.82)
        # Reder et al. (2015)
        self.watertempcorrection_FC = pcr.scalar(1.07)

        # solar radiation dependent decay: total suspended solids (Beusen et al., 2005)
        self.tss = vos.readPCRmapClone(
            options["TSSmap"],
            self.tmpDir,
            self.inputDir,
        )
        # (m2/W; Reder et al., 2015)
        self.sunlightinactivation_FC = pcr.scalar(0.0068)
        # (m-1; Reder et al., 2015)
        self.attenuation_FC = 0.0931 * self.tss + 0.881

        # sedimentation: the stream depth must exceed 0.5 m for sedimentation to occur (m)
        self.threshold_FC_settlingdepth = pcr.scalar(0.5)
        # (m/day; Reder et al., 2015)
        self.settlingvelocity_FC = pcr.scalar(1.656)

        # calculate the loadings within the model run
        if options["calculateLoads"] == "True":
            self.calculateLoads = True
            logger.info("Loadings calculated within model run.")

            if options["loadsPerSector"] == "True":
                self.loadsPerSector = True
                logger.info("Option to report loads per sector enabled")
            else:
                self.loadsPerSector = False

        else:
            self.calculateLoads = False
            self.loadsPerSector = False
            logger.info("Loadings are prescribed (i.e. akin to a forcing) to the model")

        if self.calculateLoads:

            # pollutant loading input data: domestic, gridded annual population at 5 arcmin (Lange & Geiger, 2020)

            self.PopulationNC = vos.getFullPath(options["PopulationNC"], self.inputDir)
            # average (regional) excretion rates
            self.Dom_ExcrLoadNC = vos.getFullPath(
                options["Dom_ExcrLoadNC"], self.inputDir
            )
            # (g/capita/day)
            self.DomTDS_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                self.Dom_ExcrLoadNC, "Dom_Fixed_TDSload"
            )
            # (g/capita/day)
            self.DomBOD_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                self.Dom_ExcrLoadNC, "Dom_Fixed_BODload"
            )
            # (cfu/capita/day)
            self.DomFC_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                self.Dom_ExcrLoadNC, "Dom_Fixed_FCload"
            )

            # manufacturing: average (regional) effluent concentrations
            self.Man_EfflConcNC = vos.getFullPath(
                options["Man_EfflConcNC"], self.inputDir
            )
            # (mg/L, i.e. g/m3)
            self.ManTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.Man_EfflConcNC, "Man_Fixed_TDSload"
            )
            # (mg/L, i.e. g/m3)
            self.ManBOD_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.Man_EfflConcNC, "Man_Fixed_BODload"
            )
            # (cfu/100mL)
            self.ManFC_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.Man_EfflConcNC, "Man_Fixed_FCload"
            )

            # urban surface runoff: urban fraction, from 0 (no urban) to 1 (all urban)
            self.UrbanFractionNC = vos.getFullPath(
                options["UrbanFractionNC"], self.inputDir
            )
            # average (regional) effluent concentrations
            self.USR_EfflConcNC = vos.getFullPath(
                options["USR_EfflConcNC"], self.inputDir
            )
            # (mg/L, i.e. g/m3)
            self.USRTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.USR_EfflConcNC, "USR_Fixed_TDSload"
            )
            # (mg/L, i.e. g/m3)
            self.USRBOD_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.USR_EfflConcNC, "USR_Fixed_BODload"
            )
            # (cfu/100mL)
            self.USRFC_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.USR_EfflConcNC, "USR_Fixed_FCload"
            )

            # livestock: gridded populations, 2010, 5 arcmin (Gilbert et al., 2018)
            self.LivPopulationNC = vos.getFullPath(
                options["LivPopulationNC"], self.inputDir
            )
            # average (regional) excretion rates
            self.Liv_ExcrLoadNC = vos.getFullPath(
                options["Liv_ExcrLoadNC"], self.inputDir
            )
            # (g/stock/day)
            self.Bufallo_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "bufallo_BODload"
            )
            # (g/stock/day)
            self.Chicken_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "chicken_BODload"
            )
            # (g/stock/day)
            self.Cow_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "cow_BODload"
            )
            # (g/stock/day)
            self.Duck_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "duck_BODload"
            )
            # (g/stock/day)
            self.Goat_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "goat_BODload"
            )
            # (g/stock/day)
            self.Horse_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "horse_BODload"
            )
            # (g/stock/day)
            self.Pig_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "pig_BODload"
            )
            # (g/stock/day)
            self.Sheep_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "sheep_BODload"
            )
            # (cfu/stock/day)
            self.Bufallo_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "bufallo_FCload"
            )
            # (cfu/stock/day)
            self.Chicken_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "chicken_FCload"
            )
            # (cfu/stock/day)
            self.Cow_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "cow_FCload"
            )
            # (cfu/stock/day)
            self.Duck_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "duck_FCload"
            )
            # (cfu/stock/day)
            self.Goat_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "goat_FCload"
            )
            # (cfu/stock/day)
            self.Horse_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "horse_FCload"
            )
            # (cfu/stock/day)
            self.Pig_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "pig_FCload"
            )
            # (cfu/stock/day)
            self.Sheep_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                self.Liv_ExcrLoadNC, "sheep_FCload"
            )

            # irrigation: soil concentration averaged over the topsoil and subsoil
            self.Irr_EfflConcNC = vos.getFullPath(
                options["Irr_EfflConcNC"], self.inputDir
            )
            # (mg/L)
            self.IrrTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                self.Irr_EfflConcNC, "soil_TDS"
            )

        else:
            # paths to (non-natural) TDS, BOD and FC loading inputs
            self.TDSloadNC = vos.getFullPath(options["TDSloadNC"], self.inputDir)
            self.BODloadNC = vos.getFullPath(options["BODloadNC"], self.inputDir)
            self.FCloadNC = vos.getFullPath(options["FCloadNC"], self.inputDir)

        self.getICs(options, initialConditions, channelStorage, initialWaterHeight)

    def getState(self):
        result = {}
        if self.using_qualloc:
            # TDS concentration (mg/L)
            result["salinity"] = self.salinity
            # BOD concentration (mg/L)
            result["organic"] = self.organic
            # FC concentration (cfu/100mL)
            result["pathogen"] = self.pathogen
        if self.calculateLoads:
            # average irrigation gross demand (m/day)
            result["avg_irrGrossDemand"] = self.avg_irrGrossDemand
            # average net liquid water transferred to the soil (m/day)
            result["avg_netLqWaterToSoil"] = self.avg_netLqWaterToSoil
        # water temperature (K)
        result["waterTemperature"] = self.waterTemp
        # ice thickness (m)
        result["iceThickness"] = self.iceThickness
        # routed TDS load (g), converted to salinity pollution (mg/L)
        result["routedTDS"] = self.routedTDS
        # routed BOD load (g), converted to organic pollution (mg/L)
        result["routedBOD"] = self.routedBOD
        # routed FC load (cfu), converted to pathogen pollution (cfu/100mL)
        result["routedFC"] = self.routedFC

        if self.calculateLoads and self.loadsPerSector:
            # pollutants routed per sector (for the analysis of sectoral contributions): domestic
            # (g TDS)
            result["routedDomTDS"] = self.routedDomTDS
            # (g BOD)
            result["routedDomBOD"] = self.routedDomBOD
            # (10^6 cfu)
            result["routedDomFC"] = self.routedDomFC
            # manufacturing
            # (g TDS)
            result["routedManTDS"] = self.routedManTDS
            # (g BOD)
            result["routedManBOD"] = self.routedManBOD
            # (10^6 cfu)
            result["routedManFC"] = self.routedManFC
            # urban surface runoff
            # (g TDS)
            result["routedUSRTDS"] = self.routedUSRTDS
            # (g BOD)
            result["routedUSRBOD"] = self.routedUSRBOD
            # (10^6 cfu)
            result["routedUSRFC"] = self.routedUSRFC
            # intensive livestock
            # (g BOD)
            result["routedintLivBOD"] = self.routedintLivBOD
            # (10^6 cfu)
            result["routedintLivFC"] = self.routedintLivFC
            # extensive livestock
            # (g BOD)
            result["routedextLivBOD"] = self.routedextLivBOD
            # (10^6 cfu)
            result["routedextLivFC"] = self.routedextLivFC
            # irrigation
            # (g TDS)
            result["routedIrrTDS"] = self.routedIrrTDS
        return result

    def getICs(self, options, iniConditions, channelStorage, initialWaterHeight):
        if iniConditions is None:
            if self.using_qualloc:
                # salinity pollution
                self.salinity = vos.readPCRmapClone(
                    options["salinityIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                # organic pollution
                self.organic = vos.readPCRmapClone(
                    options["organicIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                # pathogen pollution
                self.pathogen = vos.readPCRmapClone(
                    options["pathogenIni"],
                    self.tmpDir,
                    self.inputDir,
                )

            self.waterTemp = vos.readPCRmapClone(
                options["waterTemperatureIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.iceThickness = vos.readPCRmapClone(
                options["iceThicknessIni"],
                self.tmpDir,
                self.inputDir,
            )
            # salinity pollution
            self.routedTDS = vos.readPCRmapClone(
                options["routedTDSIni"],
                self.tmpDir,
                self.inputDir,
            )
            # organic pollution
            self.routedBOD = vos.readPCRmapClone(
                options["routedBODIni"],
                self.tmpDir,
                self.inputDir,
            )
            # pathogen pollution
            self.routedFC = vos.readPCRmapClone(
                options["routedFCIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial average irrigation demand and net liquid water to the soil, for the irrigation return flows
            if self.calculateLoads:
                self.avg_irrGrossDemand = vos.readPCRmapClone(
                    options["avg_irrGrossDemandIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.avg_netLqWaterToSoil = vos.readPCRmapClone(
                    options["avg_netLqWaterToSoilIni"],
                    self.tmpDir,
                    self.inputDir,
                )

                # per water quality sector
                if self.loadsPerSector:
                    self.routedDomTDS = vos.readPCRmapClone(
                        options["routedDomTDSIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedDomBOD = vos.readPCRmapClone(
                        options["routedDomBODIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedDomFC = vos.readPCRmapClone(
                        options["routedDomFCIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedManTDS = vos.readPCRmapClone(
                        options["routedManTDSIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedManBOD = vos.readPCRmapClone(
                        options["routedManBODIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedManFC = vos.readPCRmapClone(
                        options["routedManFCIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedUSRTDS = vos.readPCRmapClone(
                        options["routedUSRTDSIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedUSRBOD = vos.readPCRmapClone(
                        options["routedUSRBODIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedUSRFC = vos.readPCRmapClone(
                        options["routedUSRFCIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedintLivBOD = vos.readPCRmapClone(
                        options["routedintLivBODIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedintLivFC = vos.readPCRmapClone(
                        options["routedintLivFCIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedextLivBOD = vos.readPCRmapClone(
                        options["routedextLivBODIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedextLivFC = vos.readPCRmapClone(
                        options["routedextLivFCIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.routedIrrTDS = vos.readPCRmapClone(
                        options["routedIrrTDSIni"],
                        self.tmpDir,
                        self.inputDir,
                    )

        else:
            if self.using_qualloc:
                self.salinity = iniConditions["routing"]["salinity"]
                self.organic = iniConditions["routing"]["organic"]
                self.pathogen = iniConditions["routing"]["pathogen"]

            self.waterTemp = iniConditions["routing"]["waterTemperature"]
            self.iceThickness = iniConditions["routing"]["iceThickness"]
            self.routedTDS = iniConditions["routing"]["routedTDS"]
            self.routedBOD = iniConditions["routing"]["routedBOD"]
            self.routedFC = iniConditions["routing"]["routedFC"]

            # initial average irrigation demand and net liquid water to the soil, for the irrigation return flows
            if self.calculateLoads:
                self.avg_irrGrossDemand = iniConditions["routing"]["avg_irrGrossDemand"]
                self.avg_netLqWaterToSoil = iniConditions["routing"][
                    "avg_netLqWaterToSoil"
                ]

                # per water quality sector
                if self.loadsPerSector:
                    self.routedDomTDS = iniConditions["routing"]["routedDomTDS"]
                    self.routedDomBOD = iniConditions["routing"]["routedDomBOD"]
                    self.routedDomFC = iniConditions["routing"]["routedDomFC"]
                    self.routedManTDS = iniConditions["routing"]["routedManTDS"]
                    self.routedManBOD = iniConditions["routing"]["routedManBOD"]
                    self.routedManFC = iniConditions["routing"]["routedManFC"]
                    self.routedUSRTDS = iniConditions["routing"]["routedUSRTDS"]
                    self.routedUSRBOD = iniConditions["routing"]["routedUSRBOD"]
                    self.routedUSRFC = iniConditions["routing"]["routedUSRFC"]
                    self.routedintLivBOD = iniConditions["routing"]["routedintLivBOD"]
                    self.routedintLivFC = iniConditions["routing"]["routedintLivFC"]
                    self.routedextLivBOD = iniConditions["routing"]["routedextLivBOD"]
                    self.routedextLivFC = iniConditions["routing"]["routedextLivFC"]
                    self.routedIrrTDS = iniConditions["routing"]["routedIrrTDS"]

        if self.using_qualloc:
            self.salinity = pcr.ifthen(self.landmask, pcr.cover(self.salinity, 0.0))
            self.organic = pcr.ifthen(self.landmask, pcr.cover(self.organic, 0.0))
            self.pathogen = pcr.ifthen(self.landmask, pcr.cover(self.pathogen, 0.0))

        self.waterTemp = pcr.ifthen(self.landmask, pcr.cover(self.waterTemp, 0.0))
        self.iceThickness = pcr.ifthen(self.landmask, pcr.cover(self.iceThickness, 0.0))
        self.dissolved_oxygen = (1 - 0.0001148 * self.elevation) * pcr.exp(
            -139.34411
            + (157570.1) / (self.waterTemp)
            - (66423080.0) / (self.waterTemp**2)
            + (12438000000.0) / (self.waterTemp**3)
            - (862194900000.0) / (self.waterTemp**4)
        )
        self.channelStorageTimeBefore = channelStorage
        self.totEW = (
            channelStorage * self.waterTemp * self.specificHeatWater * self.densityWater
        )
        self.temp_water_height = initialWaterHeight
        self.routedTDS = pcr.ifthen(self.landmask, pcr.cover(self.routedTDS, 0.0))
        self.routedBOD = pcr.ifthen(self.landmask, pcr.cover(self.routedBOD, 0.0))
        self.routedFC = pcr.ifthen(self.landmask, pcr.cover(self.routedFC, 0.0))

        # per water quality sector
        if self.calculateLoads:
            self.avg_irrGrossDemand = pcr.ifthen(
                self.landmask, pcr.cover(self.avg_irrGrossDemand, 0.0)
            )
            self.avg_netLqWaterToSoil = pcr.ifthen(
                self.landmask, pcr.cover(self.avg_netLqWaterToSoil, 0.0)
            )

            if self.loadsPerSector:
                self.routedDomTDS = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedDomTDS, 0.0)
                )
                self.routedDomBOD = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedDomBOD, 0.0)
                )
                self.routedDomFC = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedDomFC, 0.0)
                )
                self.routedManTDS = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedManTDS, 0.0)
                )
                self.routedManBOD = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedManBOD, 0.0)
                )
                self.routedManFC = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedManFC, 0.0)
                )
                self.routedUSRTDS = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedUSRTDS, 0.0)
                )
                self.routedUSRBOD = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedUSRBOD, 0.0)
                )
                self.routedUSRFC = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedUSRFC, 0.0)
                )
                self.routedintLivBOD = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedintLivBOD, 0.0)
                )
                self.routedintLivFC = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedintLivFC, 0.0)
                )
                self.routedextLivBOD = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedextLivBOD, 0.0)
                )
                self.routedextLivFC = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedextLivFC, 0.0)
                )
                self.routedIrrTDS = pcr.ifthen(
                    self.landmask, pcr.cover(self.routedIrrTDS, 0.0)
                )

    def update_before_routing(
        self,
        meteo,
        landSurface,
        groundwater,
        currTimeStep,
        channelStorage,
        waterBodyStorage,
        runoff,
        dynamicFracWat,
        water_height,
        waterBodyEvaporation,
    ):
        """Returns the water body evaporation (m), set to zero where the water is frozen."""
        self.channelStorage = channelStorage
        self.runoff = runoff
        self.dynamicFracWat = dynamicFracWat
        self.water_height = water_height
        self.waterBodyEvaporation = waterBodyEvaporation

        self.waterBodyStorageTimeBefore = (
            waterBodyStorage + self.WaterBodies.waterBodyOutflow
        )

        # daily input data, unless adjusted in the function
        self.readExtensiveMeteo(currTimeStep)

        # pollutant loadings
        if self.calculateLoads:
            # daily input data for estimating the pollutant loadings, unless adjusted in the function
            self.readPollutantLoadingsInputData(currTimeStep)
            # pollutant loadings calculated (or read) daily, unless adjusted in the function
            self.calculatePollutantLoadings(currTimeStep, landSurface, groundwater)
        else:
            self.readPollutantLoadings(currTimeStep)

        # heat dumps from water temperature dependent power plants (J/day)
        self.PowTwload = pcr.cover(
            landSurface.nonIrrReturnFlowVolumePerSector["thermoelectric"]
            * self.specificHeatWater
            * self.densityWater
            * landSurface.water_demand.water_demand_thermoelectric.min_Tlmax_dTlmax,
            0.0,
        )

        self.channelStorageTimeBefore = pcr.max(0.0, self.channelStorage)
        self.qualityLocal(meteo, landSurface, groundwater, currTimeStep)
        self.qualityWaterBody()

        return self.waterBodyEvaporation

    def route_sub_step(
        self, channelStorageForRouting, subDischarge, length_of_sub_time_step
    ):
        self.subDischarge = subDischarge
        self.channelStorageNow = pcr.max(0.0, channelStorageForRouting)
        self.qualityRouting(length_of_sub_time_step)
        self.channelStorageTimeBefore = pcr.max(0.0, self.channelStorageNow)

    def update_after_routing(
        self,
        currTimeStep,
        channelStorage,
        dynamicFracWat,
        disChanWaterBody,
        avgDischarge,
        yMean,
        wMean,
    ):
        self.channelStorage = channelStorage
        self.dynamicFracWat = dynamicFracWat
        self.disChanWaterBody = disChanWaterBody
        self.avgDischarge = avgDischarge
        self.yMean = yMean
        self.wMean = wMean

        self.qualityWaterBodyAverage(currTimeStep)
        self.estimate_concentrations()

    def update_statistics(self, landSurface, timestepsToAvgDischarge):
        # average irrigation water allocated over the last 30 days (needed when loadings
        # are calculated in the loop)
        if self.calculateLoads:
            # average irrigation gross demand over the last 30 days
            irrGrossDemand = deepcopy(landSurface.irrGrossDemand)
            deltaAno_irrGrossDemand = (
                pcr.max(0.0, irrGrossDemand) - self.avg_irrGrossDemand
            )
            self.avg_irrGrossDemand = (
                self.avg_irrGrossDemand
                + deltaAno_irrGrossDemand / pcr.min(30.0, timestepsToAvgDischarge)
            )
            self.avg_irrGrossDemand = pcr.max(0.0, self.avg_irrGrossDemand)

            # average netLqWaterToSoil over the last 30 days
            deltaAno_netLqWaterToSoil = (
                pcr.max(0.0, landSurface.netLqWaterToSoil) - self.avg_netLqWaterToSoil
            )
            self.avg_netLqWaterToSoil = (
                self.avg_netLqWaterToSoil
                + deltaAno_netLqWaterToSoil / pcr.min(30.0, timestepsToAvgDischarge)
            )
            self.avg_netLqWaterToSoil = pcr.max(0.0, self.avg_netLqWaterToSoil)

    def reported_variable_names(self):
        names = [
            "waterTemp",
            "iceThickness",
            "TDSload",
            "routedTDS",
            "salinity",
            "BODload",
            "routedBOD",
            "organic",
            "dissolved_oxygen",
            "FCload",
            "routedFC",
            "pathogen",
        ]
        if self.loadsPerSector:
            for constituent, sectors in (
                ("TDS", ["Dom", "Man", "USR", "Irr"]),
                ("BOD", ["Dom", "Man", "USR", "intLiv", "extLiv"]),
                ("FC", ["Dom", "Man", "USR", "intLiv", "extLiv"]),
            ):
                for sector in sectors:
                    names.append(f"{sector}_{constituent}load")
                    names.append(f"routed{sector}{constituent}")
        return names

    def readExtensiveMeteo(self, currTimeStep):
        # read the meteorological input directly from netCDF files
        if currTimeStep.day == 1:
            self.cloudCover = vos.netcdf2PCRobjClone(
                self.cloudFileNC,
                "cld",
                str(currTimeStep.fulldate),
                useDoy="monthly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            ) / pcr.scalar(100)

            self.vaporPressure = vos.netcdf2PCRobjClone(
                self.vapFileNC,
                "vap",
                str(currTimeStep.fulldate),
                useDoy="monthly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            )

            self.annualT = vos.netcdf2PCRobjClone(
                self.annualTFileNC,
                "tas",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            ) + pcr.scalar(273.15)

        self.radiation = vos.netcdf2PCRobjClone(
            self.radFileNC,
            "rsds",
            str(currTimeStep.fulldate),
            useDoy="daily",
            LatitudeLongitude=True,
            specificFillValue=-999.0,
        )

        # atmospheric emissivity (-) from vapour pressure:
        # min(1, (0.53 + 0.0065 * vaporPressure**0.5) * (1 + 0.4 * cloudCover))
        self.atmosEmis = pcr.scalar(1.0)
        cld1 = pcr.roundoff(10 * self.cloudCover + 0.5)
        cld0 = cld1 - 1
        sun0 = pcr.lookupscalar(self.sunFracTBL, cld0)
        deltaSun = (pcr.lookupscalar(self.sunFracTBL, cld1) - sun0) / (cld1 - cld0)
        sunFrac = sun0 + (10 * self.cloudCover - cld0) * deltaSun
        radFrac = self.radCon + self.radSlope * sunFrac
        self.rsw = radFrac * self.radiation

    def readPollutantLoadingsInputData(self, currTimeStep):

        if currTimeStep.doy == 1:
            # domestic: gridded population (5 arcmin)
            self.Population = vos.netcdf2PCRobjClone(
                self.PopulationNC,
                "Population",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # urban surface runoff: urban fraction (0-1)
            self.urban_area_fraction = vos.netcdf2PCRobjClone(
                self.UrbanFractionNC,
                "urban_fraction",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            # zero urban fraction if missing (e.g. for lakes)
            self.urban_area_fraction = pcr.cover(self.urban_area_fraction, 0)

            # livestock populations (5 arcmin): buffalo
            self.BufalloPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "BufalloPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.BufalloPopulation = pcr.cover(self.BufalloPopulation, 0.0)

            # chicken
            self.ChickenPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "ChickenPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.ChickenPopulation = pcr.cover(self.ChickenPopulation, 0.0)

            # cow
            self.CowPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "CowPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.CowPopulation = pcr.cover(self.CowPopulation, 0.0)

            # duck
            self.DuckPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "DuckPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.DuckPopulation = pcr.cover(self.DuckPopulation, 0.0)

            # goat
            self.GoatPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "GoatPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.GoatPopulation = pcr.cover(self.GoatPopulation, 0.0)

            # horse
            self.HorsePopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "HorsePop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.HorsePopulation = pcr.cover(self.HorsePopulation, 0.0)

            # pig
            self.PigPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "PigPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.PigPopulation = pcr.cover(self.PigPopulation, 0.0)

            # sheep
            self.SheepPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "SheepPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.SheepPopulation = pcr.cover(self.SheepPopulation, 0.0)

            # livestock densities accounting for livestock units (Wen et al., 2018); convert m2 to km2
            self.cellArea_km2 = self.cellArea / 1000000.0
            self.LivDensityThres = pcr.scalar(25.0)

            self.BufalloDensity = self.BufalloPopulation / self.cellArea_km2
            self.ChickenDensity = (self.ChickenPopulation * 0.01) / self.cellArea_km2
            self.CowDensity = self.CowPopulation / self.cellArea_km2
            self.DuckDensity = (self.DuckPopulation * 0.01) / self.cellArea_km2
            self.GoatDensity = (self.GoatPopulation * 0.1) / self.cellArea_km2
            self.HorseDensity = self.HorsePopulation / self.cellArea_km2
            self.PigDensity = (self.PigPopulation * 0.3) / self.cellArea_km2
            self.SheepDensity = (self.SheepPopulation * 0.1) / self.cellArea_km2

            # wastewater treatment plant (point) locations and service zones
            self.WWt_plantID = pcr.nominal(
                vos.netcdf2PCRobjClone(
                    self.WWtPlantsNC,
                    "plant_id",
                    str(currTimeStep.fulldate),
                    useDoy="yearly",
                    LatitudeLongitude=True,
                    specificFillValue=None,
                )
            )
            self.WWt_zoneID = pcr.nominal(
                vos.netcdf2PCRobjClone(
                    self.WWtPlantsNC,
                    "zone_id",
                    str(currTimeStep.fulldate),
                    useDoy="yearly",
                    LatitudeLongitude=True,
                    specificFillValue=None,
                )
            )

            # wastewater treatment: fraction of wastewater that is collected and treated
            self.WWt_ct = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_ct",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # country-level fraction of uncollected wastewater with basic sanitation
            self.WWt_bs = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_bs",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # country-level fraction of uncollected wastewater that is open defecation
            self.WWt_od = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_od",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of TDS removed at each wastewater treatment plant
            self.WWt_TDS_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "TDS_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of BOD removed at each wastewater treatment plant
            self.WWt_BOD_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "BOD_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of FC removed at each wastewater treatment plant
            self.WWt_FC_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "FC_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

    def calculatePollutantLoadings(self, currTimeStep, landSurface, groundwater):
        # calculate the pollutant loadings
        logger.info("Calculating pollutant loadings")

        # fraction of direct runoff (direct runoff / total runoff), used to transport pollution from open
        # defecation and extensive livestock
        self.frac_surfaceRunoff = vos.getValDivZero(
            landSurface.directRunoff,
            (landSurface.landSurfaceRunoff + groundwater.baseflow),
        )

        # pollution from municipal wastewater (domestic, manufacturing and urban surface runoff)

        # gross domestic loadings: population (capita) * excretion rate per capita (g/capita/day; cfu/capita/day),
        # with removal at the wastewater treatment plant (g/day)
        self.Dom_TDSload = pcr.ifthenelse(
            # cells not in a wastewater treatment zone
            self.WWt_zoneID == 0,
            (self.Population * self.DomTDS_ExcrLoad * self.WWt_bs)
            + (
                self.Population
                * self.DomTDS_ExcrLoad
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID
                # accumulate the loadings over the treatment zone to the plant location
                == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.Population * self.DomTDS_ExcrLoad,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                # after accumulation, set locations without a treatment plant to 0
                0.0,
            ),
        )

        # (g/day)
        self.Dom_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            (self.Population * self.DomBOD_ExcrLoad * self.WWt_bs)
            + (
                self.Population
                * self.DomBOD_ExcrLoad
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.Population * self.DomBOD_ExcrLoad,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.Dom_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            (self.Population * (self.DomFC_ExcrLoad / 1000000.0) * self.WWt_bs)
            + (
                self.Population
                * (self.DomFC_ExcrLoad / 1000000.0)
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.Population * (self.DomFC_ExcrLoad / 1000000.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # gross manufacturing loadings: manufacturing wastewater (m3/day) * average effluent concentration (mg/L; cfu/100mL)
        if self.includeSectors["industry"]:
            self.ManWWp = landSurface.nonIrrReturnFlowVolumePerSector["industry"]
        elif self.includeSectors["manufacture"]:
            self.ManWWp = landSurface.nonIrrReturnFlowVolumePerSector["manufacture"]

        # (g/day)
        self.Man_TDSload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * self.ManTDS_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.ManWWp * self.ManTDS_EfflConc,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (g/day)
        self.Man_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * self.ManBOD_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.ManWWp * self.ManBOD_EfflConc,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.Man_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * (self.ManFC_EfflConc / 100.0),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.ManWWp * (self.ManFC_EfflConc / 100.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # gross urban surface runoff loadings: USR return flow (m3/day) * average effluent concentration (mg/L; cfu/100mL)
        # (m3/day)
        self.USR_RF = (
            landSurface.directRunoff * self.urban_area_fraction * self.cellArea
        )

        # (g/day)
        self.USR_TDSload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * self.USRTDS_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.USR_RF * self.USRTDS_EfflConc,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (g/day)
        self.USR_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * self.USRBOD_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.USR_RF * self.USRBOD_EfflConc,
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.USR_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * (self.USRFC_EfflConc / 100.0),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.USR_RF * (self.USRFC_EfflConc / 100.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # pollution from livestock (intensive and extensive): BOD
        self.intLiv_Bufallo_BODload = pcr.ifthenelse(
            self.BufalloDensity > self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_BODload,
            0.0,
        )
        self.intLiv_Chicken_BODload = pcr.ifthenelse(
            self.ChickenDensity > self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_BODload,
            0.0,
        )
        self.intLiv_Cow_BODload = pcr.ifthenelse(
            self.CowDensity > self.LivDensityThres,
            self.CowPopulation * self.Cow_BODload,
            0.0,
        )
        self.intLiv_Duck_BODload = pcr.ifthenelse(
            self.DuckDensity > self.LivDensityThres,
            self.DuckPopulation * self.Duck_BODload,
            0.0,
        )
        self.intLiv_Goat_BODload = pcr.ifthenelse(
            self.GoatDensity > self.LivDensityThres,
            self.GoatPopulation * self.Goat_BODload,
            0.0,
        )
        self.intLiv_Horse_BODload = pcr.ifthenelse(
            self.HorseDensity > self.LivDensityThres,
            self.HorsePopulation * self.Horse_BODload,
            0.0,
        )
        self.intLiv_Pig_BODload = pcr.ifthenelse(
            self.PigDensity > self.LivDensityThres,
            self.PigPopulation * self.Pig_BODload,
            0.0,
        )
        self.intLiv_Sheep_BODload = pcr.ifthenelse(
            self.SheepDensity > self.LivDensityThres,
            self.SheepPopulation * self.Sheep_BODload,
            0.0,
        )

        # (g/day)
        self.intLiv_BODload = (
            self.intLiv_Bufallo_BODload
            + self.intLiv_Chicken_BODload
            + self.intLiv_Cow_BODload
            + self.intLiv_Duck_BODload
            + self.intLiv_Goat_BODload
            + self.intLiv_Horse_BODload
            + self.intLiv_Pig_BODload
            + self.intLiv_Sheep_BODload
        )

        # (g/day)
        self.intLiv_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.intLiv_BODload * self.frac_surfaceRunoff,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.intLiv_BODload, 0),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct))
                * self.frac_surfaceRunoff,
                0.0,
            ),
        )

        self.extLiv_Bufallo_BODload = pcr.ifthenelse(
            self.BufalloDensity <= self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_BODload,
            0.0,
        )
        self.extLiv_Chicken_BODload = pcr.ifthenelse(
            self.ChickenDensity <= self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_BODload,
            0.0,
        )
        self.extLiv_Cow_BODload = pcr.ifthenelse(
            self.CowDensity <= self.LivDensityThres,
            self.CowPopulation * self.Cow_BODload,
            0.0,
        )
        self.extLiv_Duck_BODload = pcr.ifthenelse(
            self.DuckDensity <= self.LivDensityThres,
            self.DuckPopulation * self.Duck_BODload,
            0.0,
        )
        self.extLiv_Goat_BODload = pcr.ifthenelse(
            self.GoatDensity <= self.LivDensityThres,
            self.GoatPopulation * self.Goat_BODload,
            0.0,
        )
        self.extLiv_Horse_BODload = pcr.ifthenelse(
            self.HorseDensity <= self.LivDensityThres,
            self.HorsePopulation * self.Horse_BODload,
            0.0,
        )
        self.extLiv_Pig_BODload = pcr.ifthenelse(
            self.PigDensity <= self.LivDensityThres,
            self.PigPopulation * self.Pig_BODload,
            0.0,
        )
        self.extLiv_Sheep_BODload = pcr.ifthenelse(
            self.SheepDensity <= self.LivDensityThres,
            self.SheepPopulation * self.Sheep_BODload,
            0.0,
        )

        # (g/day)
        self.extLiv_BODload = (
            self.extLiv_Bufallo_BODload
            + self.extLiv_Chicken_BODload
            + self.extLiv_Cow_BODload
            + self.extLiv_Duck_BODload
            + self.extLiv_Goat_BODload
            + self.extLiv_Horse_BODload
            + self.extLiv_Pig_BODload
            + self.extLiv_Sheep_BODload
        ) * self.frac_surfaceRunoff

        # FC
        self.intLiv_Bufallo_FCload = pcr.ifthenelse(
            self.BufalloDensity > self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_FCload,
            0.0,
        )
        self.intLiv_Chicken_FCload = pcr.ifthenelse(
            self.ChickenDensity > self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_FCload,
            0.0,
        )
        self.intLiv_Cow_FCload = pcr.ifthenelse(
            self.CowDensity > self.LivDensityThres,
            self.CowPopulation * self.Cow_FCload,
            0.0,
        )
        self.intLiv_Duck_FCload = pcr.ifthenelse(
            self.DuckDensity > self.LivDensityThres,
            self.DuckPopulation * self.Duck_FCload,
            0.0,
        )
        self.intLiv_Goat_FCload = pcr.ifthenelse(
            self.GoatDensity > self.LivDensityThres,
            self.GoatPopulation * self.Goat_FCload,
            0.0,
        )
        self.intLiv_Horse_FCload = pcr.ifthenelse(
            self.HorseDensity > self.LivDensityThres,
            self.HorsePopulation * self.Horse_FCload,
            0.0,
        )
        self.intLiv_Pig_FCload = pcr.ifthenelse(
            self.PigDensity > self.LivDensityThres,
            self.PigPopulation * self.Pig_FCload,
            0.0,
        )
        self.intLiv_Sheep_FCload = pcr.ifthenelse(
            self.SheepDensity > self.LivDensityThres,
            self.SheepPopulation * self.Sheep_FCload,
            0.0,
        )

        self.intLiv_FCload = (
            self.intLiv_Bufallo_FCload
            + self.intLiv_Chicken_FCload
            + self.intLiv_Cow_FCload
            + self.intLiv_Duck_FCload
            + self.intLiv_Goat_FCload
            + self.intLiv_Horse_FCload
            + self.intLiv_Pig_FCload
            + self.intLiv_Sheep_FCload
        ) / 1000000.0

        # (million cfu/day)
        self.intLiv_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.intLiv_FCload * self.frac_surfaceRunoff,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.intLiv_FCload, 0),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct))
                * self.frac_surfaceRunoff,
                0.0,
            ),
        )

        self.extLiv_Bufallo_FCload = pcr.ifthenelse(
            self.BufalloDensity <= self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_FCload,
            0.0,
        )
        self.extLiv_Chicken_FCload = pcr.ifthenelse(
            self.ChickenDensity <= self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_FCload,
            0.0,
        )
        self.extLiv_Cow_FCload = pcr.ifthenelse(
            self.CowDensity <= self.LivDensityThres,
            self.CowPopulation * self.Cow_FCload,
            0.0,
        )
        self.extLiv_Duck_FCload = pcr.ifthenelse(
            self.DuckDensity <= self.LivDensityThres,
            self.DuckPopulation * self.Duck_FCload,
            0.0,
        )
        self.extLiv_Goat_FCload = pcr.ifthenelse(
            self.GoatDensity <= self.LivDensityThres,
            self.GoatPopulation * self.Goat_FCload,
            0.0,
        )
        self.extLiv_Horse_FCload = pcr.ifthenelse(
            self.HorseDensity <= self.LivDensityThres,
            self.HorsePopulation * self.Horse_FCload,
            0.0,
        )
        self.extLiv_Pig_FCload = pcr.ifthenelse(
            self.PigDensity <= self.LivDensityThres,
            self.PigPopulation * self.Pig_FCload,
            0.0,
        )
        self.extLiv_Sheep_FCload = pcr.ifthenelse(
            self.SheepDensity <= self.LivDensityThres,
            self.SheepPopulation * self.Sheep_FCload,
            0.0,
        )

        # (million cfu/day)
        self.extLiv_FCload = (
            (
                self.extLiv_Bufallo_FCload
                + self.extLiv_Chicken_FCload
                + self.extLiv_Cow_FCload
                + self.extLiv_Duck_FCload
                + self.extLiv_Goat_FCload
                + self.extLiv_Horse_FCload
                + self.extLiv_Pig_FCload
                + self.extLiv_Sheep_FCload
            )
            / 1000000.0
        ) * self.frac_surfaceRunoff

        # pollution from irrigation: irrigation return flow (m3/day) * soil TDS (mg/L); the return flows
        # follow from direct runoff, interflow and baseflow
        self.irr_rf_from_direct_runoff = landSurface.directRunoff * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )
        self.irr_rf_from_interflow = landSurface.interflowTotal * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )
        self.irr_rf_from_baseflow = groundwater.baseflow * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )

        # total irrigation return flow (m3/day)
        self.Irr_RF = (
            self.irr_rf_from_direct_runoff
            + self.irr_rf_from_interflow
            + self.irr_rf_from_baseflow
        ) * self.cellArea
        # (g/day)
        self.Irr_TDSload = self.Irr_RF * self.IrrTDS_EfflConc

        # pollution from all human activities combined (g/day)
        self.TDSload = (
            self.Dom_TDSload + self.Man_TDSload + self.USR_TDSload + self.Irr_TDSload
        )
        # (g/day)
        self.BODload = (
            self.Dom_BODload
            + self.Man_BODload
            + self.USR_BODload
            + self.intLiv_BODload
            + self.extLiv_BODload
        )
        # (million cfu/day)
        self.FCload = (
            self.Dom_FCload
            + self.Man_FCload
            + self.USR_FCload
            + self.intLiv_FCload
            + self.extLiv_FCload
        )

    def readPollutantLoadings(self, currTimeStep):
        # use precalculated pollutant loadings
        logger.info("Reading loadings directly")

        # TDS loadings (all sectors combined)
        self.TDSload = vos.netcdf2PCRobjClone(
            self.TDSloadNC,
            "TDSload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.TDSload = pcr.ifthen(self.landmask, self.TDSload)

        # BOD loadings (all sectors combined)
        self.BODload = vos.netcdf2PCRobjClone(
            self.BODloadNC,
            "BODload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.BODload = pcr.ifthen(self.landmask, self.BODload)

        # FC loadings (all sectors combined)
        self.FCload = vos.netcdf2PCRobjClone(
            self.FCloadNC,
            "FCload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.FCload = pcr.ifthen(self.landmask, self.FCload)

    def qualityLocal(
        self, meteo, landSurface, groundwater, currTimeStep, timeSec=vos.secondsPerDay()
    ):

        # water temperature: surface water energy fluxes (W/m2) within the current time step; ice formation
        # is evaluated before routing to account for the loss in water height, the vertical energy change is
        # evaluated and warming is capped at the air temperature

        # constants and variables
        self.temperatureKelvin = meteo.temperature + pcr.scalar(273.15)
        landRunoff = self.runoff
        self.correctPrecip = pcr.scalar(0.0)
        self.dynamicFracWatBeforeRouting = self.dynamicFracWat

        # land temperature considering the runoff fractions
        landT = pcr.cover(
            landSurface.directRunoff
            / landRunoff
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin - self.deltaTPrec)
            + landSurface.interflowTotal
            / landRunoff
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin)
            + groundwater.baseflow
            / landRunoff
            * pcr.max(self.iceThresTemp + 5.0, self.annualT),
            self.temperatureKelvin,
        )

        # heat transfer (ice and water)
        iceHeatTransfer = self.heatTransferWater * (
            self.temperatureKelvin - self.iceThresTemp
        )
        waterHeatTransfer = self.heatTransferIce * (self.iceThresTemp - self.waterTemp)

        # check for ice
        noIce = pcr.ifthenelse(
            self.iceThickness > 0,
            pcr.boolean(0),
            pcr.ifthenelse(
                ((iceHeatTransfer - waterHeatTransfer) < 0)
                & (self.temperatureKelvin < self.iceThresTemp),
                pcr.boolean(0),
                pcr.boolean(1),
            ),
        )

        # update the water heat transfer based on the presence of ice
        waterHeatTransfer = pcr.ifthenelse(
            noIce,
            self.heatTransferWater * (self.temperatureKelvin - self.waterTemp),
            waterHeatTransfer,
        )

        # radiative heat transfer
        radiativHeatTransfer = (
            1 - pcr.ifthenelse(noIce, self.albedoWater, self.albedoSnow)
        ) * self.rsw
        radiativHeatTransfer -= self.stefanBoltzman * (
            pcr.ifthenelse(noIce, self.waterTemp, self.iceThresTemp) ** 4
            - self.atmosEmis * self.temperatureKelvin**4
        )

        # advected energy due to precipitation and inflow
        advectedEnergyPrecip = (
            pcr.max(0, self.correctPrecip)
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin - self.deltaTPrec)
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )
        advectedEnergyInflow = (
            (1 - self.dynamicFracWat)
            / self.dynamicFracWat
            * landRunoff
            * landT
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )
        advectedEnergyInflow -= (
            landSurface.actSurfaceWaterAbstract
            / self.dynamicFracWat
            * self.waterTemp
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )

        # ice formation and thickness
        diceHeatTransfer = pcr.ifthenelse(
            noIce, 0, iceHeatTransfer - waterHeatTransfer + radiativHeatTransfer
        )
        self.deltaIceThickness = (
            -diceHeatTransfer * timeSec / (self.densityWater * self.latentHeatFusion)
        )
        self.deltaIceThickness = pcr.max(-self.iceThickness, self.deltaIceThickness)
        self.deltaIceThickness = pcr.min(
            self.deltaIceThickness, pcr.max(0, self.maxIceThickness - self.iceThickness)
        )

        # direct gain over the water surface
        watQ = pcr.ifthenelse(
            self.temperatureKelvin >= self.iceThresTemp,
            pcr.max(0, self.correctPrecip)
            - self.waterBodyEvaporation / self.dynamicFracWat,
            0,
        )
        # TODO: move
        self.waterBodyEvaporation = pcr.ifthenelse(
            self.temperatureKelvin >= self.iceThresTemp,
            self.waterBodyEvaporation,
            0,
        )

        # vertical gains/losses
        deltaIceThickness_melt = pcr.max(0, -self.deltaIceThickness)
        verticalGain = (
            watQ
            + (landRunoff - landSurface.actSurfaceWaterAbstract) / (self.dynamicFracWat)
            + deltaIceThickness_melt
        )

        # water storage change
        dtotStorLoc = verticalGain
        totStorLoc = (
            self.channelStorageTimeBefore / (self.dynamicFracWat * self.cellArea)
            - dtotStorLoc
        )
        totStorLoc = (
            self.WaterBodies.return_water_body_storage_to_channel(
                self.channelStorageTimeBefore
            )
            / (self.dynamicFracWat * self.cellArea)
            - dtotStorLoc
        )

        # energy in the channel
        self.totEW = (
            totStorLoc * self.waterTemp * self.specificHeatWater * self.densityWater
        )
        dtotStorLoc = pcr.max(-totStorLoc, dtotStorLoc)

        # latent heat flux and advected energy of water evaporation
        latentHeat = (
            -self.waterBodyEvaporation
            / self.dynamicFracWat
            * self.densityWater
            * self.latentHeatVapor
            / timeSec
        )
        advectedEnergyPrecip += (
            deltaIceThickness_melt
            * self.iceThresTemp
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )

        # total energy change
        totEWC = totStorLoc * self.specificHeatWater * self.densityWater
        dtotEWC = dtotStorLoc * self.specificHeatWater * self.densityWater
        dtotEWLoc = (
            waterHeatTransfer + pcr.scalar(noIce) * (radiativHeatTransfer + latentHeat)
        ) * timeSec
        dtotEWAdv = (advectedEnergyInflow + advectedEnergyPrecip) * timeSec

        dtotEWLoc = pcr.min(
            dtotEWLoc,
            pcr.max(0, totEWC * self.temperatureKelvin - self.totEW)
            + pcr.ifthenelse(
                dtotStorLoc > 0,
                pcr.max(0, dtotEWC * self.temperatureKelvin - dtotEWAdv),
                0,
            ),
        )
        dtotEWLoc = pcr.ifthenelse(
            self.waterTemp > self.temperatureKelvin, pcr.min(0, dtotEWLoc), dtotEWLoc
        )
        dtotEWLoc = pcr.max(
            dtotEWLoc,
            pcr.min(
                0,
                (totEWC + dtotEWC)
                * pcr.max(self.temperatureKelvin, self.iceThresTemp + 0.1)
                - (self.totEW + dtotEWAdv),
            ),
        )

        # update the water height, energy and water temperature
        self.temp_water_height = pcr.max(1e-16, totStorLoc + dtotStorLoc)
        self.totEW = pcr.max(0, self.totEW + dtotEWLoc + dtotEWAdv)

        # temporary fix for an error in the water temperature in the first time step
        if currTimeStep.timeStepPCR != 1:
            self.waterTemp = pcr.ifthenelse(
                self.temp_water_height > self.critical_water_height,
                self.totEW
                / self.temp_water_height
                / (self.specificHeatWater * self.densityWater),
                self.temperatureKelvin,
            )
        self.waterTemp = pcr.min(
            pcr.ifthenelse(
                self.waterTemp < self.iceThresTemp + 0.1,
                self.iceThresTemp + 0.1,
                self.waterTemp,
            ),
            self.maxThresTemp,
        )
        # water temperature (degC)
        self.waterTemp_C = self.waterTemp - pcr.scalar(273.15)

        # BOD (non-conservative; depends on water temperature only): temperature-dependent decay
        self.BODdecay_temperature = pcr.cover(
            self.k_BOD * (self.watertempcorrection_BOD ** (self.waterTemp_C - 20)), 0.0
        )

        # FC (non-conservative; depends on temperature, solar radiation and sedimentation): temperature-dependent decay
        self.FCdecay_temperature = pcr.cover(
            self.darkinactivation_FC
            * (self.watertempcorrection_FC ** (self.waterTemp_C - 20)),
            0.0,
        )

        # solar radiation dependent decay, with a minimum water depth of 0.1 m for the decay coefficients
        self.water_height_pathogen = pcr.max(self.water_height, 0.1)
        self.FCdecay_solarradiation = pcr.cover(
            self.sunlightinactivation_FC
            * (self.rsw / (self.attenuation_FC * self.water_height_pathogen))
            * (1 - (pcr.exp(-(self.attenuation_FC * self.water_height_pathogen)))),
            0.0,
        )

        # sedimentation (day-1)
        self.FCdecay_sedimentation = pcr.ifthenelse(
            self.water_height_pathogen > self.threshold_FC_settlingdepth,
            pcr.cover(self.settlingvelocity_FC / self.water_height, 0.0),
            0,
        )

    def qualityRouting(self, timeSec):

        channelTransFrac = pcr.cover(
            pcr.max(
                pcr.min(
                    (self.subDischarge * timeSec) / self.channelStorageTimeBefore,
                    1.0,
                ),
                0.0,
            ),
            0.0,
        )

        # energy (water temperature) routing, adding heat effluents from power plants (J per timeSec)
        self.volumeEW = self.volumeEW + (
            self.PowTwload * (timeSec / vos.secondsPerDay())
        )
        dtotEWLat = channelTransFrac * self.volumeEW
        self.volumeEW = self.volumeEW + pcr.upstream(self.lddMap, dtotEWLat) - dtotEWLat

        # salinity (TDS) routing
        self.routedTDS = self.routedTDS + (
            self.TDSload * (timeSec / vos.secondsPerDay())
        )
        dTDSLat = channelTransFrac * self.routedTDS
        self.routedTDS = self.routedTDS + pcr.upstream(self.lddMap, dTDSLat) - dTDSLat

        if self.loadsPerSector:
            self.routedDomTDS = self.routedDomTDS + (
                self.Dom_TDSload * (timeSec / vos.secondsPerDay())
            )
            dDomTDSLat = channelTransFrac * self.routedDomTDS
            self.routedDomTDS = (
                self.routedDomTDS + pcr.upstream(self.lddMap, dDomTDSLat) - dDomTDSLat
            )

            self.routedManTDS = self.routedManTDS + (
                self.Man_TDSload * (timeSec / vos.secondsPerDay())
            )
            dManTDSLat = channelTransFrac * self.routedManTDS
            self.routedManTDS = (
                self.routedManTDS + pcr.upstream(self.lddMap, dManTDSLat) - dManTDSLat
            )

            self.routedUSRTDS = self.routedUSRTDS + (
                self.USR_TDSload * (timeSec / vos.secondsPerDay())
            )
            dUSRTDSLat = channelTransFrac * self.routedUSRTDS
            self.routedUSRTDS = (
                self.routedUSRTDS + pcr.upstream(self.lddMap, dUSRTDSLat) - dUSRTDSLat
            )

            self.routedIrrTDS = self.routedIrrTDS + (
                self.Irr_TDSload * (timeSec / vos.secondsPerDay())
            )
            dIrrTDSLat = channelTransFrac * self.routedIrrTDS
            self.routedIrrTDS = (
                self.routedIrrTDS + pcr.upstream(self.lddMap, dIrrTDSLat) - dIrrTDSLat
            )

        # organic (BOD) routing
        self.routedBOD = self.routedBOD + (
            self.BODload * (timeSec / vos.secondsPerDay())
        )
        dBODLat = channelTransFrac * self.routedBOD
        self.BODdecay = pcr.exp(
            -(self.BODdecay_temperature) * (timeSec / vos.secondsPerDay())
        )
        self.routedBOD = (
            self.routedBOD + pcr.upstream(self.lddMap, dBODLat) - dBODLat
        ) * self.BODdecay

        if self.loadsPerSector:
            self.routedDomBOD = self.routedDomBOD + (
                self.Dom_BODload * (timeSec / vos.secondsPerDay())
            )
            dDomBODLat = channelTransFrac * self.routedDomBOD
            self.routedDomBOD = (
                self.routedDomBOD + pcr.upstream(self.lddMap, dDomBODLat) - dDomBODLat
            ) * self.BODdecay

            self.routedManBOD = self.routedManBOD + (
                self.Man_BODload * (timeSec / vos.secondsPerDay())
            )
            dManBODLat = channelTransFrac * self.routedManBOD
            self.routedManBOD = (
                self.routedManBOD + pcr.upstream(self.lddMap, dManBODLat) - dManBODLat
            ) * self.BODdecay

            self.routedUSRBOD = self.routedUSRBOD + (
                self.USR_BODload * (timeSec / vos.secondsPerDay())
            )
            dUSRBODLat = channelTransFrac * self.routedUSRBOD
            self.routedUSRBOD = (
                self.routedUSRBOD + pcr.upstream(self.lddMap, dUSRBODLat) - dUSRBODLat
            ) * self.BODdecay

            self.routedintLivBOD = self.routedintLivBOD + (
                self.intLiv_BODload * (timeSec / vos.secondsPerDay())
            )
            dintLivBODLat = channelTransFrac * self.routedintLivBOD
            self.routedintLivBOD = (
                self.routedintLivBOD
                + pcr.upstream(self.lddMap, dintLivBODLat)
                - dintLivBODLat
            ) * self.BODdecay

            self.routedextLivBOD = self.routedextLivBOD + (
                self.extLiv_BODload * (timeSec / vos.secondsPerDay())
            )
            dextLivBODLat = channelTransFrac * self.routedextLivBOD
            self.routedextLivBOD = (
                self.routedextLivBOD
                + pcr.upstream(self.lddMap, dextLivBODLat)
                - dextLivBODLat
            ) * self.BODdecay

        # pathogen (FC) routing
        self.routedFC = self.routedFC + (self.FCload * (timeSec / vos.secondsPerDay()))
        dFCLat = channelTransFrac * self.routedFC
        self.FCdecay = pcr.exp(
            -(
                self.FCdecay_temperature
                + self.FCdecay_solarradiation
                + self.FCdecay_sedimentation
            )
            * (timeSec / vos.secondsPerDay())
        )
        self.routedFC = (
            self.routedFC + pcr.upstream(self.lddMap, dFCLat) - dFCLat
        ) * self.FCdecay

        if self.loadsPerSector:
            self.routedDomFC = self.routedDomFC + (
                self.Dom_FCload * (timeSec / vos.secondsPerDay())
            )
            dDomFCLat = channelTransFrac * self.routedDomFC
            self.routedDomFC = (
                self.routedDomFC + pcr.upstream(self.lddMap, dDomFCLat) - dDomFCLat
            ) * self.FCdecay

            self.routedManFC = self.routedManFC + (
                self.Man_FCload * (timeSec / vos.secondsPerDay())
            )
            dManFCLat = channelTransFrac * self.routedManFC
            self.routedManFC = (
                self.routedManFC + pcr.upstream(self.lddMap, dManFCLat) - dManFCLat
            ) * self.FCdecay

            self.routedUSRFC = self.routedUSRFC + (
                self.USR_FCload * (timeSec / vos.secondsPerDay())
            )
            dUSRFCLat = channelTransFrac * self.routedUSRFC
            self.routedUSRFC = (
                self.routedUSRFC + pcr.upstream(self.lddMap, dUSRFCLat) - dUSRFCLat
            ) * self.FCdecay

            self.routedintLivFC = self.routedintLivFC + (
                self.intLiv_FCload * (timeSec / vos.secondsPerDay())
            )
            dintLivFCLat = channelTransFrac * self.routedintLivFC
            self.routedintLivFC = (
                self.routedintLivFC
                + pcr.upstream(self.lddMap, dintLivFCLat)
                - dintLivFCLat
            ) * self.FCdecay

            self.routedextLivFC = self.routedextLivFC + (
                self.extLiv_FCload * (timeSec / vos.secondsPerDay())
            )
            dextLivFCLat = channelTransFrac * self.routedextLivFC
            self.routedextLivFC = (
                self.routedextLivFC
                + pcr.upstream(self.lddMap, dextLivFCLat)
                - dextLivFCLat
            ) * self.FCdecay

    def qualityWaterBody(self):

        lakeTransFrac = pcr.max(
            pcr.min(
                (self.WaterBodies.waterBodyOutflow) / (self.waterBodyStorageTimeBefore),
                1.0,
            ),
            0.0,
        )
        lakeTransFrac = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, lakeTransFrac), 0.0
        )

        # water temperature (energy in the water body)
        energyTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(
                        self.landmask,
                        self.totEW * self.dynamicFracWat * self.cellArea,
                    ),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.totEW * self.dynamicFracWat * self.cellArea,
        )
        self.volumeEW = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                energyTotal * lakeTransFrac,
            ),
            energyTotal,
        )
        self.remainingVolumeEW = pcr.cover(
            pcr.ifthen(
                self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * energyTotal
            ),
            0.0,
        )

        # salinity (TDS in the water body)
        wbTDSTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedTDS),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedTDS,
        )
        self.routedTDS = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbTDSTotal * lakeTransFrac,
            ),
            wbTDSTotal,
        )
        self.wbRemainingTDS = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbTDSTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomTDS,
            )
            self.routedDomTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomTDSTotal * lakeTransFrac,
                ),
                wbDomTDSTotal,
            )
            self.wbRemainingDomTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomTDSTotal
                ),
                0.0,
            )

            wbManTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManTDS,
            )
            self.routedManTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManTDSTotal * lakeTransFrac,
                ),
                wbManTDSTotal,
            )
            self.wbRemainingManTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManTDSTotal
                ),
                0.0,
            )

            wbUSRTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRTDS,
            )
            self.routedUSRTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRTDSTotal * lakeTransFrac,
                ),
                wbUSRTDSTotal,
            )
            self.wbRemainingUSRTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRTDSTotal
                ),
                0.0,
            )

            wbIrrTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedIrrTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedIrrTDS,
            )
            self.routedIrrTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbIrrTDSTotal * lakeTransFrac,
                ),
                wbIrrTDSTotal,
            )
            self.wbRemainingIrrTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbIrrTDSTotal
                ),
                0.0,
            )

        # organic (BOD in the water body)
        wbBODTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedBOD),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedBOD,
        )
        self.routedBOD = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbBODTotal * lakeTransFrac,
            ),
            wbBODTotal,
        )
        self.wbRemainingBOD = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbBODTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomBOD,
            )
            self.routedDomBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomBODTotal * lakeTransFrac,
                ),
                wbDomBODTotal,
            )
            self.wbRemainingDomBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomBODTotal
                ),
                0.0,
            )

            wbManBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManBOD,
            )
            self.routedManBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManBODTotal * lakeTransFrac,
                ),
                wbManBODTotal,
            )
            self.wbRemainingManBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManBODTotal
                ),
                0.0,
            )

            wbUSRBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRBOD,
            )
            self.routedUSRBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRBODTotal * lakeTransFrac,
                ),
                wbUSRBODTotal,
            )
            self.wbRemainingUSRBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRBODTotal
                ),
                0.0,
            )

            wbintLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedintLivBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivBOD,
            )
            self.routedintLivBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbintLivBODTotal * lakeTransFrac,
                ),
                wbintLivBODTotal,
            )
            self.wbRemainingintLivBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut,
                    (1 - lakeTransFrac) * wbintLivBODTotal,
                ),
                0.0,
            )

            wbextLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedextLivBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivBOD,
            )
            self.routedextLivBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbextLivBODTotal * lakeTransFrac,
                ),
                wbextLivBODTotal,
            )
            self.wbRemainingextLivBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut,
                    (1 - lakeTransFrac) * wbextLivBODTotal,
                ),
                0.0,
            )

        # pathogen (FC in the water body)
        wbFCTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedFC),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedFC,
        )
        self.routedFC = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbFCTotal * lakeTransFrac,
            ),
            wbFCTotal,
        )
        self.wbRemainingFC = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbFCTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomFC,
            )
            self.routedDomFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomFCTotal * lakeTransFrac,
                ),
                wbDomFCTotal,
            )
            self.wbRemainingDomFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomFCTotal
                ),
                0.0,
            )

            wbManFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManFC,
            )
            self.routedManFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManFCTotal * lakeTransFrac,
                ),
                wbManFCTotal,
            )
            self.wbRemainingManFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManFCTotal
                ),
                0.0,
            )

            wbUSRFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRFC,
            )
            self.routedUSRFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRFCTotal * lakeTransFrac,
                ),
                wbUSRFCTotal,
            )
            self.wbRemainingUSRFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRFCTotal
                ),
                0.0,
            )

            wbintLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedintLivFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivFC,
            )
            self.routedintLivFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbintLivFCTotal * lakeTransFrac,
                ),
                wbintLivFCTotal,
            )
            self.wbRemainingintLivFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbintLivFCTotal
                ),
                0.0,
            )

            wbextLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedextLivFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivFC,
            )
            self.routedextLivFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbextLivFCTotal * lakeTransFrac,
                ),
                wbextLivFCTotal,
            )
            self.wbRemainingextLivFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbextLivFCTotal
                ),
                0.0,
            )

    def qualityWaterBodyAverage(self, currTimeStep):

        # water temperature (energy averaged over the water body)
        self.totalVolumeEW = self.volumeEW + self.remainingVolumeEW

        energyTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.totalVolumeEW),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.totalVolumeEW,
        )
        energyAverageLakeCell = pcr.cover(
            energyTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            energyTotal,
        )
        self.totEW = pcr.cover(
            energyAverageLakeCell / (self.dynamicFracWat * self.cellArea), 1e-16
        )

        self.temp_water_height = self.WaterBodies.return_water_body_storage_to_channel(
            self.channelStorageNow
        ) / (self.dynamicFracWat * self.cellArea)

        iceReductionFactor = pcr.ifthen(
            self.landmask,
            pcr.cover(self.dynamicFracWatBeforeRouting / self.dynamicFracWat, 1.0),
        )

        self.deltaIceThickness = iceReductionFactor * self.deltaIceThickness
        self.deltaIceThickness = pcr.min(self.deltaIceThickness, self.temp_water_height)

        self.iceThickness = iceReductionFactor * self.iceThickness
        self.iceThickness = pcr.max(
            0,
            self.iceThickness
            + (
                self.deltaIceThickness
                + pcr.ifthenelse(
                    self.temperatureKelvin >= self.iceThresTemp, 0, self.correctPrecip
                )
            ),
        )
        self.iceThickness = pcr.ifthenelse(
            (self.iceThickness <= 0.001) & (self.deltaIceThickness < 0),
            0,
            self.iceThickness,
        )
        self.iceThickness = pcr.min(self.iceThickness, self.maxIceThickness)

        self.channelStorageNow = (
            self.channelStorageNow
            - self.deltaIceThickness * self.dynamicFracWat * self.cellArea
        )

        if currTimeStep.timeStepPCR == 1:
            logger.info(
                "Issue with estimating water temperature in first timestep: retrieving water temperature from initial condition"
            )
        else:
            self.waterTemp = pcr.ifthenelse(
                self.temp_water_height > self.critical_water_height,
                self.totEW
                / self.temp_water_height
                / (self.specificHeatWater * self.densityWater),
                self.temperatureKelvin,
            )

        self.waterTemp = pcr.min(
            pcr.ifthenelse(
                self.waterTemp < self.iceThresTemp + 0.1,
                self.iceThresTemp + 0.1,
                self.waterTemp,
            ),
            self.maxThresTemp,
        )
        # water temperature (degC)
        self.waterTemp_C = self.waterTemp - pcr.scalar(273.15)

        # salinity (TDS averaged over the water body)
        wbTDSTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedTDS + self.wbRemainingTDS),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedTDS,
        )
        self.routedTDS = pcr.cover(
            wbTDSTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbTDSTotal,
        )

        if self.loadsPerSector:
            wbDomTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomTDS + self.wbRemainingDomTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomTDS,
            )
            self.routedDomTDS = pcr.cover(
                wbDomTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomTDSTotal,
            )

            wbManTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManTDS + self.wbRemainingManTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManTDS,
            )
            self.routedManTDS = pcr.cover(
                wbManTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManTDSTotal,
            )

            wbUSRTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRTDS + self.wbRemainingUSRTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRTDS,
            )
            self.routedUSRTDS = pcr.cover(
                wbUSRTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRTDSTotal,
            )

            wbIrrTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedIrrTDS + self.wbRemainingIrrTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedIrrTDS,
            )
            self.routedIrrTDS = pcr.cover(
                wbIrrTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbIrrTDSTotal,
            )

        # organic (BOD averaged over the water body)
        wbBODTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedBOD + self.wbRemainingBOD),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedBOD,
        )
        self.routedBOD = pcr.cover(
            wbBODTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbBODTotal,
        )

        if self.loadsPerSector:
            wbDomBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomBOD + self.wbRemainingDomBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomBOD,
            )
            self.routedDomBOD = pcr.cover(
                wbDomBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomBODTotal,
            )

            wbManBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManBOD + self.wbRemainingManBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManBOD,
            )
            self.routedManBOD = pcr.cover(
                wbManBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManBODTotal,
            )

            wbUSRBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRBOD + self.wbRemainingUSRBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRBOD,
            )
            self.routedUSRBOD = pcr.cover(
                wbUSRBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRBODTotal,
            )

            wbintLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedintLivBOD + self.wbRemainingintLivBOD,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivBOD,
            )
            self.routedintLivBOD = pcr.cover(
                wbintLivBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbintLivBODTotal,
            )

            wbextLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedextLivBOD + self.wbRemainingextLivBOD,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivBOD,
            )
            self.routedextLivBOD = pcr.cover(
                wbextLivBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbextLivBODTotal,
            )

        # pathogen (FC averaged over the water body)
        wbFCTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedFC + self.wbRemainingFC),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedFC,
        )
        self.routedFC = pcr.cover(
            wbFCTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbFCTotal,
        )

        if self.loadsPerSector:
            wbDomFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomFC + self.wbRemainingDomFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomFC,
            )
            self.routedDomFC = pcr.cover(
                wbDomFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomFCTotal,
            )

            wbManFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManFC + self.wbRemainingManFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManFC,
            )
            self.routedManFC = pcr.cover(
                wbManFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManFCTotal,
            )

            wbUSRFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRFC + self.wbRemainingUSRFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRFC,
            )
            self.routedUSRFC = pcr.cover(
                wbUSRFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRFCTotal,
            )

            wbintLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedintLivFC + self.wbRemainingintLivFC,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivFC,
            )
            self.routedintLivFC = pcr.cover(
                wbintLivFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbintLivFCTotal,
            )

            wbextLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedextLivFC + self.wbRemainingextLivFC,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivFC,
            )
            self.routedextLivFC = pcr.cover(
                wbextLivFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbextLivFCTotal,
            )

    def estimate_concentrations(self):

        # channel storage with discharge threshold, to estimate in-stream concentrations
        self.channelStorage_Qthres = pcr.ifthenelse(
            pcr.cover(self.disChanWaterBody, vos.MV) > self.WQ_discharge_threshold,
            self.channelStorage,
            vos.MV,
        )

        # total dissolved solids concentration, salinity indicator (mg/L)
        self.salinity = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            (self.routedTDS / self.channelStorage_Qthres) + self.backgroundSalinity,
            vos.MV,
        )

        # biochemical oxygen demand concentration, organic indicator (mg/L)
        self.organic = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            self.routedBOD / self.channelStorage_Qthres,
            vos.MV,
        )

        # dissolved oxygen concentration (Streeter-Phelps equation)
        self.k1 = self.BODdecay_temperature * self.organic
        # oxygen saturation (mg/L)
        self.DOsat = (1 - 0.0001148 * self.elevation) * pcr.exp(
            -139.34411
            + (157570.1) / (self.waterTemp)
            - (66423080.0) / (self.waterTemp**2)
            + (12438000000.0) / (self.waterTemp**3)
            - (862194900000.0) / (self.waterTemp**4)
        )
        # velocity assuming a rectangular channel (m/s)
        self.velocity = self.avgDischarge / (self.yMean * self.wMean)
        # reaeration rate (day-1; O'Connor and Dobbins, 1958)
        self.k2 = 3.93 * (self.velocity**0.5) / (self.yMean**1.5)
        self.k2 = pcr.ifthenelse(
            self.k2 > 1.5, 1.5, pcr.ifthenelse(self.k2 < 0.4, 0.4, self.k2)
        )
        self.dissolved_oxygen = pcr.ifthenelse(
            self.dissolved_oxygen
            - self.k1
            + self.k2 * (self.DOsat - self.dissolved_oxygen)
            < 0.0,
            0.0,
            self.dissolved_oxygen
            - self.k1
            + self.k2 * (self.DOsat - self.dissolved_oxygen),
        )
        # dissolved oxygen concentration (mg/L)
        self.dissolved_oxygen = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV, self.dissolved_oxygen, vos.MV
        )

        # fecal coliform concentration, pathogen indicator (cfu/100mL)
        self.pathogen = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            self.routedFC * 100.0 / self.channelStorage_Qthres,
            vos.MV,
        )
