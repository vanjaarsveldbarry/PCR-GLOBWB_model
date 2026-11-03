import logging

import pcraster as pcr

from common import virtualOS as vos
from pcrglobwb import groundwater, landSurface, meteo, routing

logger = logging.getLogger(__name__)


class PCRGlobWB(object):

    def __init__(self, configuration, currTimeStep, initialState=None):
        self._modelTime = currTimeStep

        vos.set_clone(configuration.cloneMap)

        self.lddMap = vos.readPCRmapClone(
            configuration.routingOptions["lddMap"],
            absolutePath=configuration.globalOptions["inputDir"],
            isLddMap=True,
        )
        # make sure the ldd map is correct and of type ldd
        self.lddMap = pcr.lddrepair(pcr.ldd(self.lddMap))

        if configuration.globalOptions["landmask"] != "None":
            self.landmask = vos.readPCRmapClone(
                configuration.globalOptions["landmask"],
                absolutePath=configuration.globalOptions["inputDir"],
            )
        else:
            self.landmask = pcr.defined(self.lddMap)

        self.meteo = meteo.Meteo(configuration, self.landmask, initialState)
        self.landSurface = landSurface.LandSurface(
            configuration, self.landmask, initialState
        )
        self.groundwater = groundwater.Groundwater(
            configuration, self.landmask, initialState
        )
        self.routing = routing.Routing(configuration, initialState, self.lddMap)

    def getState(self):
        result = {}

        result["meteo"] = self.meteo.getState()

        result["landSurface"] = self.landSurface.getState()
        result["groundwater"] = self.groundwater.getState()
        result["routing"] = self.routing.getState()

        return result

    def update(self):
        logger.info("Updating model for time %s", self._modelTime)

        self.meteo.update(self.routing, self._modelTime)
        self.landSurface.update(
            self.meteo, self.groundwater, self.routing, self._modelTime
        )
        self.groundwater.update(self.landSurface, self.routing, self._modelTime)
        self.routing.update(
            self.landSurface, self.groundwater, self._modelTime, self.meteo
        )
