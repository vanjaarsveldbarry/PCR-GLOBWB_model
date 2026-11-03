import logging

import pcraster as pcr

from common import utils

logger = logging.getLogger(__name__)


class WaterBalance(object):

    def __init__(self, model, modelTime):
        self._modelTime = modelTime

        self.landmask = model.landmask
        self.meteo = model.meteo
        self.landSurface = model.landSurface
        self.groundwater = model.groundwater
        self.routing = model.routing

    def begin_step(self):
        self._landWaterStoresAtBeginning = self.totalLandWaterStores()
        self._surfaceWaterStoresAtBeginning = self.totalSurfaceWaterStores()

    def end_step(self):
        landWaterStoresAtEnd = self.totalLandWaterStores()
        surfaceWaterStoresAtEnd = self.totalSurfaceWaterStores()

        self.checkLandSurfaceWaterBalance(
            self._landWaterStoresAtBeginning, landWaterStoresAtEnd
        )

        # TODO: include water balance checks for the surface water part and for land surface and surface water combined

        self.report_summary(
            self._landWaterStoresAtBeginning,
            landWaterStoresAtEnd,
            self._surfaceWaterStoresAtBeginning,
            surfaceWaterStoresAtEnd,
        )

    def report_summary(
        self,
        landWaterStoresAtBeginning,
        landWaterStoresAtEnd,
        surfaceWaterStoresAtBeginning,
        surfaceWaterStoresAtEnd,
    ):

        # reset the totals on the first day of the year
        if self._modelTime.doy == 1 or self._modelTime.isFirstTimestep():

            self.precipitationAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.list_of_land_surface_variables = self.landSurface.fluxVars + [
                "desalinationAbstraction",
                "desalinationAllocation",
                "actSurfaceWaterAbstract",
                "allocSurfaceWaterAbstract",
                "nonFossilGroundwaterAbs",
                "allocNonFossilGroundwater",
                "fossilGroundwaterAbstr",
                "fossilGroundwaterAlloc",
                "totalGroundwaterAbstraction",
                "totalGroundwaterAllocation",
                "nonIrrReturnFlow",
            ]

            for var in self.list_of_land_surface_variables:
                vars(self)[f"{var}Acc"] = pcr.ifthen(
                    self.landmask, pcr.spatial(pcr.scalar(0.0))
                )

            self.baseflowAcc = pcr.ifthen(self.landmask, pcr.spatial(pcr.scalar(0.0)))

            self.surfaceWaterInfAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.runoffAcc = pcr.ifthen(self.landmask, pcr.spatial(pcr.scalar(0.0)))
            self.unmetDemandAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.waterBalanceAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.absWaterBalanceAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # non-irrigation water use (m)
            self.nonIrrigationWaterUseAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # non-irrigation return flow to water bodies and water body evaporation (m)
            self.nonIrrReturnFlowAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.waterBodyEvaporationAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # surface water input/loss volume (m3) and outgoing volume at pits (m3)
            self.surfaceWaterInputAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.dischargeAtPitAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # storages at the first day of the year (or first time step): land surface (m)
            self.storageAtFirstDay = pcr.ifthen(
                self.landmask, landWaterStoresAtBeginning
            )
            # channel storage (m3)
            self.channelVolumeAtFirstDay = pcr.ifthen(
                self.landmask, surfaceWaterStoresAtBeginning
            )

        # accumulate until the last day of the year
        self.precipitationAcc += self.meteo.precipitation
        for var in self.list_of_land_surface_variables:
            vars(self)[f"{var}Acc"] += vars(self.landSurface)[var]

        self.baseflowAcc += self.groundwater.baseflow

        self.surfaceWaterInfAcc += self.groundwater.surfaceWaterInf

        self.runoffAcc += self.routing.runoff
        self.unmetDemandAcc += self.groundwater.unmetDemand

        self.waterBalance = (
            landWaterStoresAtBeginning
            - landWaterStoresAtEnd
            + self.meteo.precipitation
            + self.landSurface.irrigationWaterWithdrawal / self.routing.cellArea
            + self.groundwater.surfaceWaterInf
            - self.landSurface.actualET
            - self.routing.runoff
            - self.groundwater.nonFossilGroundwaterAbs
        )

        self.waterBalanceAcc += self.waterBalance
        self.absWaterBalanceAcc += pcr.abs(self.waterBalance)

        # consumptive water use for non-irrigation demand (m)
        self.nonIrrigationWaterUseAcc += self.landSurface.nonIrrWaterConsumption

        self.waterBodyEvaporationAcc += self.routing.waterBodyEvaporation

        # (m3)
        self.surfaceWaterInputAcc += self.routing.local_input_to_surface_water
        # (m3)
        self.dischargeAtPitAcc += self.routing.outgoing_volume_at_pits

        if self._modelTime.isLastDayOfYear() or self._modelTime.isLastTimeStep():

            logger.info("")
            # TODO: improve these water balance checks
            logger.info("Summary excludes lake, reservoir and channel storage")

            totalCellArea = utils.getMapTotal(
                pcr.ifthen(self.landmask, self.routing.cellArea)
            )
            logger.info(f"Total area = {totalCellArea / 1000000.0:e} km2")

            deltaStorageOneYear = utils.getMapVolume(
                pcr.ifthen(self.landmask, landWaterStoresAtBeginning)
                - pcr.ifthen(self.landmask, self.storageAtFirstDay),
                self.routing.cellArea,
            )
            logger.info(
                "Delta total storage days 1 to %i in %i = %e km3 = %e mm",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                deltaStorageOneYear / 1e9,
                deltaStorageOneYear * 1000 / totalCellArea,
            )

            variableList = [
                "precipitation",
                "baseflow",
                "surfaceWaterInf",
                "runoff",
                "unmetDemand",
            ]
            variableList += self.list_of_land_surface_variables

            variableList += ["waterBalance", "absWaterBalance", "nonIrrigationWaterUse"]

            for var in variableList:
                volume = utils.getMapVolume(
                    self.__getattribute__(f"{var}Acc"), self.routing.cellArea
                )

                # TODO: the calculation does not always start on day 1
                logger.info(
                    "Accumulated %s days 1 to %i in %i = %e km3 = %e mm",
                    var,
                    int(self._modelTime.doy),
                    int(self._modelTime.year),
                    volume / 1e9,
                    volume * 1000 / totalCellArea,
                )

            logger.info("")
            logger.info("The following summary is for surface water bodies.")

            deltaChannelStorageOneYear = utils.getMapTotal(
                pcr.ifthen(self.landmask, surfaceWaterStoresAtEnd)
                - pcr.ifthen(self.landmask, self.channelVolumeAtFirstDay)
            )
            logger.info(
                "Delta surface water storage days 1 to %i in %i = %e km3 = %e mm",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                deltaChannelStorageOneYear / 1e9,
                deltaChannelStorageOneYear * 1000 / totalCellArea,
            )

            variableList = ["waterBodyEvaporation"]
            for var in variableList:
                volume = utils.getMapVolume(
                    self.__getattribute__(f"{var}Acc"), self.routing.cellArea
                )
                logger.info(
                    "Accumulated %s days 1 to %i in %i = %e km3 = %e mm",
                    var,
                    int(self._modelTime.doy),
                    int(self._modelTime.year),
                    volume / 1e9,
                    volume * 1000 / totalCellArea,
                )

            surfaceWaterInputTotal = utils.getMapTotal(self.surfaceWaterInputAcc)
            logger.info(
                "Accumulated %s days 1 to %i in %i = %e km3 = %e mm",
                "surfaceWaterInput",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                surfaceWaterInputTotal / 1e9,
                surfaceWaterInputTotal * 1000 / totalCellArea,
            )

            dischargeAtPitTotal = utils.getMapTotal(self.dischargeAtPitAcc)
            logger.info(
                "Accumulated %s days 1 to %i in %i = %e km3 = %e mm",
                "dischargeAtPitTotal",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                dischargeAtPitTotal / 1e9,
                dischargeAtPitTotal * 1000 / totalCellArea,
            )

            surfaceWaterBalance = (
                deltaChannelStorageOneYear
                - surfaceWaterInputTotal
                + dischargeAtPitTotal
            )
            logger.info(
                "Accumulated %s days 1 to %i in %i = %e km3 = %e mm",
                "surfaceWaterBalance",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                surfaceWaterBalance / 1e9,
                surfaceWaterBalance * 1000 / totalCellArea,
            )

    def totalLandWaterStores(self):
        # (m), excluding surface water bodies

        if self.landSurface.numberOfSoilLayers == 2:
            total = (
                self.landSurface.interceptStor
                + self.landSurface.snowFreeWater
                + self.landSurface.snowCoverSWE
                + self.landSurface.topWaterLayer
                + self.landSurface.storUpp
                + self.landSurface.storLow
                + self.groundwater.storGroundwater
            )

        if self.landSurface.numberOfSoilLayers == 3:
            total = (
                self.landSurface.interceptStor
                + self.landSurface.snowFreeWater
                + self.landSurface.snowCoverSWE
                + self.landSurface.topWaterLayer
                + self.landSurface.storUpp000005
                + self.landSurface.storUpp005030
                + self.landSurface.storLow030150
                + self.groundwater.storGroundwater
            )

        total = pcr.ifthen(self.landmask, total)

        return total

    def totalSurfaceWaterStores(self):
        # (m3), surface water bodies only

        return pcr.ifthen(self.landmask, self.routing.channelStorage)

    def checkLandSurfaceWaterBalance(self, storesAtBeginning, storesAtEnd):

        # all stores (snow, interception, soil, groundwater), excluding routing: incoming fluxes (m)
        precipitation = pcr.ifthen(self.landmask, self.meteo.precipitation)
        satisfiedIrrGrossDemand = pcr.ifthen(
            self.landmask,
            self.landSurface.irrigationWaterWithdrawal / self.routing.cellArea,
        )
        surfaceWaterInf = pcr.ifthen(self.landmask, self.groundwater.surfaceWaterInf)
        # outgoing fluxes (m)
        actualET = pcr.ifthen(self.landmask, self.landSurface.actualET)
        runoff = pcr.ifthen(self.landmask, self.routing.runoff)
        nonFossilGroundwaterAbs = pcr.ifthen(
            self.landmask, self.groundwater.nonFossilGroundwaterAbs
        )
        # added by Joren; TODO: add an option to switch between the new and old module
        transportVolSnow = self.landSurface.transportVolSnow / self.routing.cellArea
        incomingVolSnow = self.landSurface.incomingVolSnow / self.routing.cellArea

        incomingFreeWater = self.landSurface.incomingFreeWater / self.routing.cellArea
        transportFreeWater = self.landSurface.transportFreeWater / self.routing.cellArea

        utils.waterBalanceCheck(
            [
                precipitation,
                surfaceWaterInf,
                satisfiedIrrGrossDemand,
                incomingVolSnow,
                incomingFreeWater,
            ],
            [
                actualET,
                runoff,
                nonFossilGroundwaterAbs,
                transportVolSnow,
                transportFreeWater,
            ],
            [storesAtBeginning],
            [storesAtEnd],
            "all stores (snow + interception + soil + groundwater), but except river/routing",
            True,
            self._modelTime.fulldate,
            threshold=1e-3,
        )
