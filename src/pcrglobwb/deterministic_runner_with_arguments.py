import argparse
import logging
import os
import sys

from pcraster.framework import DynamicFramework, DynamicModel

from pcrglobwb.common import disclaimer
from pcrglobwb.common.currTimeStep import ModelTime
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
        spinUpRun=False,
    ):
        DynamicModel.__init__(self)

        self.modelTime = modelTime
        self.model = PCRGlobWB(configuration, modelTime, initialState, spinUpRun)
        self.reporting = Reporting(configuration, self.model, modelTime)

        self.configuration = configuration

    def initial(self):
        pass

    def dynamic(self):

        # update the model time from the current PCRaster time step
        self.modelTime.update(self.currentTimeStep())

        # read the forcing (uses the current model time)
        self.model.read_forcings()

        # update the model
        self.model.update(report_water_balance=True)

        self.reporting.report()


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--ini", required=True, type=os.path.abspath)
    parser.add_argument("--debug", action="store_true")

    # Each option's metavar is also its legacy placeholder, replaced in the ini text in
    # this order; its dest is its key in [runArguments].
    parser.add_argument(
        "--mod",
        dest="outputDir",
        metavar="MAIN_OUTPUT_DIR",
        help="output directory; without it the ini is used as is",
    )
    parser.add_argument(
        "--mid",
        dest="inputDir",
        metavar="MAIN_INPUT_DIR",
        help="input directory",
    )
    parser.add_argument(
        "--sd",
        dest="startTime",
        metavar="START_DATE",
        help="start date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--ed",
        dest="endTime",
        metavar="END_DATE",
        help="end date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--misd",
        dest="initialStateDir",
        metavar="MAIN_INITIAL_STATE_FOLDER",
        help="main folder for all initial states",
    )
    parser.add_argument(
        "--dfis",
        dest="initialStateDate",
        metavar="DATE_FOR_INITIAL_STATES",
        help="date of all initial state files",
    )
    parser.add_argument(
        "--clone_connections",
        dest="cloneConnections",
        metavar="CLONE_CONNECTIONS",
    )
    parser.add_argument(
        "--pff",
        dest="precipitationNC",
        metavar="PRECIPITATION_FORCING_FILE",
        help="precipitation forcing file",
    )
    parser.add_argument(
        "--tff",
        dest="temperatureNC",
        metavar="TEMPERATURE_FORCING_FILE",
        help="temperature forcing file",
    )
    parser.add_argument(
        "--rpetff",
        dest="refETPotFileNC",
        metavar="REF_POT_ET_FORCING_FILE",
        help="reference potential evapotranspiration forcing file",
    )
    parser.add_argument(
        "--presff",
        dest="atmospheric_pressure",
        metavar="PRESSURE_FORCING_FILE",
        help="atmospheric pressure forcing file",
    )
    parser.add_argument(
        "--windff",
        dest="wind_speed_10m",
        metavar="WIND_FORCING_FILE",
        help="10 m wind speed forcing file",
    )
    parser.add_argument(
        "--swradff",
        dest="shortwave_radiation",
        metavar="SHORTWAVE_RADIATION_FORCING_FILE",
        help="shortwave radiation forcing file",
    )
    parser.add_argument(
        "--relhumff",
        dest="relative_humidity",
        metavar="RELATIVE_HUMIDITY_FORCING_FILE",
        help="relative humidity forcing file",
    )
    parser.add_argument(
        "--bfexp",
        dest="baseflowExponent",
        metavar="BASEFLOW_EXP_INPUT",
        help="groundwater baseflow exponent",
    )
    parser.add_argument(
        "--num_of_sp_years",
        dest="maxSpinUpsInYears",
        metavar="NUMBER_OF_SPINUP_YEARS",
        help="number of spin-up years",
    )
    parser.add_argument(
        "--clonemap",
        dest="cloneMapPrefix",
        metavar="CLONEMAP",
        help="clone map path without the .clone.map / .mask.map suffix",
    )
    parser.add_argument(
        "--use_max_fossil_gw_ini",
        dest="useMaximumStorGroundwaterFossilIni",
        metavar="USE_MAXIMUM_STOR_GROUNDWATER_FOSSIL_INI",
    )
    parser.add_argument(
        "--est_stor_gw_from_rch",
        dest="estimateStorGroundwaterIniFromRecharge",
        metavar="ESTIMATE_STOR_GROUNDWATER_INI_FROM_RECHARGE",
    )
    parser.add_argument(
        "--day_gw_rch_ini",
        dest="dailyGroundwaterRechargeIni",
        metavar="DAILY_GROUNDWATER_RECHARGE_INI",
    )
    parser.add_argument(
        "--qcf",
        dest="configuration_file_for_qualloc",
        metavar="QUALLOC_CONFIG_FILE",
        help="QUAlloc configuration file",
    )
    return parser


def ini_substitutions(args):
    """Return (run_arguments, placeholders) for Configuration; both None without --mod."""

    if args.outputDir is None:
        return None, None

    run_arguments = {}
    placeholders = {}
    for action in build_parser()._actions:
        if not action.option_strings or action.metavar is None:
            continue
        value = getattr(args, action.dest)
        if value is None:
            continue
        run_arguments[action.dest] = value
        placeholders[action.metavar] = value
        print(
            f"'{action.dest}' is set based on the system argument "
            f"({action.option_strings[0]}): {value}"
        )
    return run_arguments, placeholders


def main():

    disclaimer.print_disclaimer()

    args = build_parser().parse_args()
    run_arguments, placeholders = ini_substitutions(args)

    configuration = Configuration(
        iniFileName=args.ini,
        debug_mode=args.debug,
        system_arguments=sys.argv,
        run_arguments=run_arguments,
        placeholders=placeholders,
    )

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
                configuration, currTimeStep, initial_state, spinUpRun=True
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

    # run the model (excluding the DA scheme)
    currTimeStep.getStartEndTimeSteps(
        configuration.globalOptions["startTime"], configuration.globalOptions["endTime"]
    )

    logger.info("Transient simulation run started.")
    deterministic_runner = DeterministicRunner(
        configuration, currTimeStep, initial_state, spinUpRun=False
    )

    dynamic_framework = DynamicFramework(
        deterministic_runner, currTimeStep.nrOfTimeSteps
    )
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()


if __name__ == "__main__":
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())
