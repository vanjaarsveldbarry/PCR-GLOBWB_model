import logging
import os
import sys

import pcraster as pcr
from pcraster.framework import DynamicFramework, DynamicModel

from common import disclaimer, utils
from common import virtualOS as vos
from common.currTimeStep import ModelTime
from pcrglobwb.configuration import Configuration
from pcrglobwb.pcrglobwb import PCRGlobWB
from pcrglobwb.reporting import Reporting
from pcrglobwb.spinUp import SpinUp

logger = logging.getLogger(__name__)


class DeterministicRunner(DynamicModel):

    def __init__(
        self,
        configuration,
        modelTime,
        initialState=None,
        system_argument=None,
        spinUpRun=False,
    ):
        DynamicModel.__init__(self)

        self.modelTime = modelTime
        self.model = PCRGlobWB(configuration, modelTime, initialState, spinUpRun)
        self.reporting = Reporting(configuration, self.model, modelTime)

        # the model parameters may be modified
        self.parameter_adjusment = False
        if ("-adjparm" in list(system_argument)) or (
            "prefactorOptions" in configuration.allSections
        ):
            self.adusting_parameters(configuration, system_argument)
            self.parameter_adjusment = True

        self.configuration = configuration

    def adusting_parameters(self, configuration, system_argument):

        # global pre-multipliers given as arguments
        if "-adjparm" in list(system_argument):

            # start index of the adjustment arguments
            sai = system_argument.index("-adjparm")

            logger.info("Adjusting model parameters from the system arguments")

            # pre-multipliers for minSoilDepthFrac (linear), kSat (log), recessionCoeff (log), storCap
            # (linear) and degreeDayFactor (linear)
            multiplier_for_minSoilDepthFrac = float(system_argument[sai + 1])
            multiplier_for_kSat = float(system_argument[sai + 2])
            multiplier_for_recessionCoeff = float(system_argument[sai + 3])
            multiplier_for_storCap = float(system_argument[sai + 4])
            multiplier_for_degreeDayFactor = float(system_argument[sai + 5])

            # pre-multiplier for the reference potential ET (linear)
            self.multiplier_for_refPotET = float(system_argument[sai + 6])

            # pre-multiplier for manningsN (linear)
            multiplier_for_manningsN = float(system_argument[sai + 7])

            storGroundwaterIni_file = str(system_argument[sai + 8])

        # pre-factors can also be defined in the ini file; these overwrite the pre-multipliers
        if "prefactorOptions" in configuration.allSections:

            logger.info("Adjusting model parameters from the ini file")

            # linear; note: does not work for the changing WMIN or Joyce land cover options
            self.multiplier_for_refPotET = float(
                configuration.prefactorOptions["linear_multiplier_for_refPotET"]
            )
            # linear
            multiplier_for_degreeDayFactor = float(
                configuration.prefactorOptions["linear_multiplier_for_degreeDayFactor"]
            )
            # linear
            multiplier_for_minSoilDepthFrac = float(
                configuration.prefactorOptions["linear_multiplier_for_minSoilDepthFrac"]
            )
            # log
            multiplier_for_kSat = float(
                configuration.prefactorOptions["log_10_multiplier_for_kSat"]
            )
            # linear
            multiplier_for_storCap = float(
                configuration.prefactorOptions["linear_multiplier_for_storCap"]
            )
            # log
            multiplier_for_recessionCoeff = float(
                configuration.prefactorOptions["log_10_multiplier_for_recessionCoeff"]
            )
            # linear
            multiplier_for_manningsN = float(
                configuration.prefactorOptions["multiplier_for_manningsN"]
            )

            # file location (use the full path)
            storGroundwaterIni_file = str(
                configuration.prefactorOptions["storGroundwaterIni_file"]
            )

        # log the global pre-multipliers
        msg = (
            "\n\nMultiplier values used: \n"
            f"For minSoilDepthFrac           : {multiplier_for_minSoilDepthFrac}\n"
            f"For kSat (log-scale)           : {multiplier_for_kSat}\n"
            f"For recessionCoeff (log-scale) : {multiplier_for_recessionCoeff}\n"
            f"For storCap                    : {multiplier_for_storCap}\n"
            f"For degreeDayFactor            : {multiplier_for_degreeDayFactor}\n"
            f"For refPotET                   : {self.multiplier_for_refPotET}\n"
            f"For multiplier_for_manningsN   : {multiplier_for_manningsN}\n"
            f"For storGroundwaterIni_file    : {storGroundwaterIni_file}\n"
        )
        logger.info(msg)
        # and write them to a text file in the "maps" folder of outputDir (the cwd, see configuration.py)
        with open("multiplier.txt", "w") as f:
            f.write(msg)

        if storGroundwaterIni_file != "Default":
            self.model.groundwater.storGroundwater = vos.readPCRmapClone(
                storGroundwaterIni_file, configuration.tmpDir
            )
            self.model.groundwater.storGroundwater = pcr.ifthen(
                self.landmask, pcr.cover(self.model.groundwater.storGroundwater, 0.0)
            )
        pcr.report(self.model.groundwater.storGroundwater, "storGroundwaterIni.map")

        # adjust the parameters with the pre-multipliers and save the adjusted maps to the "maps"
        # folder of outputDir (the cwd, see configuration.py)
        # manningsN: minimum zero, log scale
        self.model.routing.manningsN = (
            multiplier_for_manningsN * self.model.routing.manningsN
        )
        pcr.report(self.model.routing.manningsN, "manningsN.map")

        # recessionCoeff: minimum zero, log scale
        self.model.groundwater.recessionCoeff = pcr.max(
            0.0,
            (10 ** (multiplier_for_recessionCoeff))
            * self.model.groundwater.recessionCoeff,
        )
        self.model.groundwater.recessionCoeff = pcr.min(
            1.0, self.model.groundwater.recessionCoeff
        )
        pcr.report(self.model.groundwater.recessionCoeff, "recessionCoeff.map")

        for coverType in self.model.landSurface.coverTypes:

            self.model.landSurface.landCoverObj[coverType].degreeDayFactor = pcr.max(
                0.0,
                multiplier_for_degreeDayFactor
                * self.model.landSurface.landCoverObj[coverType].degreeDayFactor,
            )
            pcraster_filename = f"degreeDayFactor_{coverType}.map"
            pcr.report(
                self.model.landSurface.landCoverObj[coverType].degreeDayFactor,
                pcraster_filename,
            )

            # kSat and storCap for the 2-layer model
            if self.model.landSurface.numberOfSoilLayers == 2:

                # kSat: minimum zero, log scale
                self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp = (
                    pcr.max(
                        0.0,
                        (10 ** (multiplier_for_kSat))
                        * self.model.landSurface.landCoverObj[
                            coverType
                        ].parameters.kSatUpp,
                    )
                )
                self.model.landSurface.landCoverObj[coverType].parameters.kSatLow = (
                    pcr.max(
                        0.0,
                        (10 ** (multiplier_for_kSat))
                        * self.model.landSurface.landCoverObj[
                            coverType
                        ].parameters.kSatLow,
                    )
                )
                pcraster_filename = f"kSatUpp_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp,
                    pcraster_filename,
                )
                pcraster_filename = f"kSatLow_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].parameters.kSatLow,
                    pcraster_filename,
                )

                # storCap: minimum zero
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp = (
                    pcr.max(
                        0.0,
                        multiplier_for_storCap
                        * self.model.landSurface.landCoverObj[
                            coverType
                        ].parameters.storCapUpp,
                    )
                )
                self.model.landSurface.landCoverObj[coverType].parameters.storCapLow = (
                    pcr.max(
                        0.0,
                        multiplier_for_storCap
                        * self.model.landSurface.landCoverObj[
                            coverType
                        ].parameters.storCapLow,
                    )
                )
                pcraster_filename = f"storCapUpp_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp,
                    pcraster_filename,
                )
                pcraster_filename = f"storCapLow_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapLow,
                    pcraster_filename,
                )

            # kSat and storCap for the 3-layer model
            if self.model.landSurface.numberOfSoilLayers == 3:

                # kSat: minimum zero, log scale
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.kSatUpp000005 = pcr.max(
                    0.0,
                    (10 ** (multiplier_for_kSat))
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatUpp000005,
                )
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.kSatUpp005030 = pcr.max(
                    0.0,
                    (10 ** (multiplier_for_kSat))
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatUpp005030,
                )
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.kSatLow030150 = pcr.max(
                    0.0,
                    (10 ** (multiplier_for_kSat))
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatLow030150,
                )
                pcraster_filename = f"kSatUpp000005_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatUpp000005,
                    pcraster_filename,
                )
                pcraster_filename = f"kSatUpp005030_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatUpp005030,
                    pcraster_filename,
                )
                pcraster_filename = f"kSatLow030150_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.kSatLow030150,
                    pcraster_filename,
                )

                # storCap: minimum zero
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.storCapUpp000005 = pcr.max(
                    0.0,
                    multiplier_for_storCap
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp000005,
                )
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.storCapUpp005030 = pcr.max(
                    0.0,
                    multiplier_for_storCap
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp005030,
                )
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.storCapLow030150 = pcr.max(
                    0.0,
                    multiplier_for_storCap
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapLow030150,
                )
                pcraster_filename = f"storCapUpp000005_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp000005,
                    pcraster_filename,
                )
                pcraster_filename = f"storCapUpp005030_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp005030,
                    pcraster_filename,
                )
                pcraster_filename = f"storCapLow030150_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapLow030150,
                    pcraster_filename,
                )

            # recalculate rootZoneWaterStorageCap (WMAX in the oldcalc script) after modifying storCap
            if self.model.landSurface.numberOfSoilLayers == 2:
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.rootZoneWaterStorageCap = (
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp
                    + self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapLow
                )
            if self.model.landSurface.numberOfSoilLayers == 3:
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.rootZoneWaterStorageCap = (
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp000005
                    + self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapUpp005030
                    + self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.storCapLow030150
                )
            pcraster_filename = f"rootZoneWaterStorageCap_{coverType}.map"
            pcr.report(
                self.model.landSurface.landCoverObj[
                    coverType
                ].parameters.rootZoneWaterStorageCap,
                pcraster_filename,
            )

            if multiplier_for_minSoilDepthFrac != 1.0:

                # minimum zero
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = (
                    pcr.max(
                        0.0,
                        multiplier_for_minSoilDepthFrac
                        * self.model.landSurface.landCoverObj[
                            coverType
                        ].minSoilDepthFrac,
                    )
                )
                # limited by maxSoilDepthFrac
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = (
                    pcr.min(
                        self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac,
                        self.model.landSurface.landCoverObj[coverType].maxSoilDepthFrac,
                    )
                )
                # maximum 1.0
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = (
                    pcr.min(
                        1.0,
                        self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac,
                    )
                )
                pcraster_filename = f"minSoilDepthFrac_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac,
                    pcraster_filename,
                )

                # recalculate arnoBeta after modifying minSoilDepthFrac
                self.model.landSurface.landCoverObj[coverType].arnoBeta = pcr.max(
                    0.001,
                    (
                        self.model.landSurface.landCoverObj[coverType].maxSoilDepthFrac
                        - 1.0
                    )
                    / (
                        1.0
                        - self.model.landSurface.landCoverObj[
                            coverType
                        ].minSoilDepthFrac
                    )
                    + self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.orographyBeta
                    - 0.01,
                )
                self.model.landSurface.landCoverObj[coverType].arnoBeta = pcr.cover(
                    pcr.max(
                        0.001, self.model.landSurface.landCoverObj[coverType].arnoBeta
                    ),
                    0.001,
                )
                pcraster_filename = f"arnoBeta_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].arnoBeta,
                    pcraster_filename,
                )

                # recalculate rootZoneWaterStorageMin (WMIN in the oldcalc script: minimum local soil
                # water capacity within the cell, in m) after modifying minSoilDepthFrac
                self.model.landSurface.landCoverObj[
                    coverType
                ].rootZoneWaterStorageMin = (
                    self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac
                    * self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.rootZoneWaterStorageCap
                )
                pcraster_filename = f"rootZoneWaterStorageMin_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].rootZoneWaterStorageMin,
                    pcraster_filename,
                )

                # recalculate rootZoneWaterStorageRange (WMAX - WMIN, in m) after modifying storCap and minSoilDepthFrac
                self.model.landSurface.landCoverObj[
                    coverType
                ].rootZoneWaterStorageRange = (
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].parameters.rootZoneWaterStorageCap
                    - self.model.landSurface.landCoverObj[
                        coverType
                    ].rootZoneWaterStorageMin
                )
                pcraster_filename = f"rootZoneWaterStorageRange_{coverType}.map"
                pcr.report(
                    self.model.landSurface.landCoverObj[
                        coverType
                    ].rootZoneWaterStorageRange,
                    pcraster_filename,
                )

    def initial(self):
        pass

    def dynamic(self):

        # update the model time from the current PCRaster time step
        self.modelTime.update(self.currentTimeStep())

        # read the forcing (uses the current model time)
        self.model.read_forcings()

        # adjust the reference potential ET with the pre-multiplier
        if self.parameter_adjusment:
            self.model.meteo.referencePotET = (
                self.model.meteo.referencePotET * self.multiplier_for_refPotET
            )

        # update the model
        self.model.update(report_water_balance=True)

        self.reporting.report()


def main():

    disclaimer.print_disclaimer()

    iniFileName = os.path.abspath(sys.argv[1])

    # modify the ini file and save it to a new location
    if "-mod" in sys.argv:
        iniFileName = utils.modify_ini_file(
            original_ini_file=iniFileName, system_argument=sys.argv
        )

    debug_mode = False
    if len(sys.argv) > 2:
        if (
            sys.argv[2] == "debug"
            or sys.argv[2] == "debug_parallel"
            or sys.argv[2] == "debug-parallel"
        ):
            debug_mode = True

    this_run_is_part_of_a_set_of_parallel_run = False
    if len(sys.argv) > 2:
        if (
            sys.argv[2] == "parallel"
            or sys.argv[2] == "debug_parallel"
            or sys.argv[2] == "debug-parallel"
        ):
            this_run_is_part_of_a_set_of_parallel_run = True

    configuration = Configuration(
        iniFileName=iniFileName, debug_mode=debug_mode, no_modification=False
    )

    # a parallel run (e.g. 5 and 6 arcmin runs) gets a specific directory based on the clone code:
    if this_run_is_part_of_a_set_of_parallel_run:
        # modify outputDir, clone map, landmask, etc. based on the command-line arguments
        clone_code = str(sys.argv[3])
        output_folder_with_clone_code = f"M{int(clone_code):02}"
        configuration.globalOptions["outputDir"] += f"/{output_folder_with_clone_code}"
        configuration.globalOptions["cloneMap"] = configuration.globalOptions[
            "cloneMap"
        ] % (int(clone_code))
        # landmask for the model calculation
        if configuration.globalOptions["landmask"] != "None":
            configuration.globalOptions["landmask"] = configuration.globalOptions[
                "landmask"
            ] % (int(clone_code))

    configuration.set_configuration(system_arguments=sys.argv)

    # time step info: year, month, day, doy, etc.
    currTimeStep = ModelTime()

    spin_up = SpinUp(configuration)

    # spin-up
    noSpinUps = int(configuration.globalOptions["maxSpinUpsInYears"])
    initial_state = None
    if noSpinUps > 0:

        logger.info(f"Spin-Up #Total Years: {noSpinUps}")

        spinUpRun = 0
        has_converged = False
        while spinUpRun < noSpinUps and not has_converged:
            spinUpRun += 1
            currTimeStep.getStartEndTimeStepsForSpinUp(
                configuration.globalOptions["startTime"], spinUpRun, noSpinUps
            )
            logger.info(f"Spin-Up Run No. {spinUpRun}")
            deterministic_runner = DeterministicRunner(
                configuration, currTimeStep, initial_state, sys.argv, spinUpRun=True
            )

            all_state_begin = deterministic_runner.model.getAllState()

            dynamic_framework = DynamicFramework(
                deterministic_runner, currTimeStep.nrOfTimeSteps
            )
            dynamic_framework.setQuiet(True)
            dynamic_framework.run()

            all_state_end = deterministic_runner.model.getAllState()

            has_converged = spin_up.checkConvergence(
                all_state_begin,
                all_state_end,
                spinUpRun,
                deterministic_runner.model.routing.cellArea,
            )

            initial_state = deterministic_runner.model.getState()

        # TODO: for a parallel run, merge when the spin-up is done and keep the states in a separate folder

    # run the model (excluding the DA scheme)
    currTimeStep.getStartEndTimeSteps(
        configuration.globalOptions["startTime"], configuration.globalOptions["endTime"]
    )

    logger.info("Transient simulation run started.")
    deterministic_runner = DeterministicRunner(
        configuration, currTimeStep, initial_state, sys.argv, spinUpRun=False
    )

    dynamic_framework = DynamicFramework(
        deterministic_runner, currTimeStep.nrOfTimeSteps
    )
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()


if __name__ == "__main__":
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())
