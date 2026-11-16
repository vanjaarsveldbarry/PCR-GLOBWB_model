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
    # The positional forms keep the old `pcrglobwb-run <ini> [debug]` calls working.
    parser.add_argument("ini_positional", nargs="?", metavar="INI")
    parser.add_argument("debug_word", nargs="?", choices=["debug"], metavar="debug")
    parser.add_argument("--ini")
    parser.add_argument("--debug", action="store_true")

    run_arguments = parser.add_argument_group("run arguments")
    parser.run_arguments = run_arguments
    run_arguments.add_argument(
        "--mod",
        dest="outputDir",
        help="output directory",
    )
    run_arguments.add_argument(
        "--mid",
        dest="inputDir",
        help="input directory",
    )
    run_arguments.add_argument(
        "--sd",
        dest="startTime",
        help="start date, YYYY-MM-DD",
    )
    run_arguments.add_argument(
        "--ed",
        dest="endTime",
        help="end date, YYYY-MM-DD",
    )
    run_arguments.add_argument(
        "--misd",
        dest="initialStateDir",
        help="main folder for all initial states",
    )
    run_arguments.add_argument(
        "--dfis",
        dest="initialStateDate",
        help="date of all initial state files",
    )
    run_arguments.add_argument(
        "--clone_connections",
        dest="cloneConnections",
    )
    run_arguments.add_argument(
        "--pff",
        dest="precipitationNC",
        help="precipitation forcing file",
    )
    run_arguments.add_argument(
        "--tff",
        dest="temperatureNC",
        help="temperature forcing file",
    )
    run_arguments.add_argument(
        "--rpetff",
        dest="refETPotFileNC",
        help="reference potential evapotranspiration forcing file",
    )
    run_arguments.add_argument(
        "--presff",
        dest="atmospheric_pressure",
        help="atmospheric pressure forcing file",
    )
    run_arguments.add_argument(
        "--windff",
        dest="wind_speed_10m",
        help="10 m wind speed forcing file",
    )
    run_arguments.add_argument(
        "--swradff",
        dest="shortwave_radiation",
        help="shortwave radiation forcing file",
    )
    run_arguments.add_argument(
        "--relhumff",
        dest="relative_humidity",
        help="relative humidity forcing file",
    )
    run_arguments.add_argument(
        "--bfexp",
        dest="baseflowExponent",
        help="groundwater baseflow exponent",
    )
    run_arguments.add_argument(
        "--num_of_sp_years",
        dest="maxSpinUpsInYears",
        help="number of spin-up years",
    )
    run_arguments.add_argument(
        "--clonemap",
        dest="cloneMapPrefix",
        help="clone map path without the .clone.map / .mask.map suffix",
    )
    run_arguments.add_argument(
        "--use_max_fossil_gw_ini",
        dest="useMaximumStorGroundwaterFossilIni",
    )
    run_arguments.add_argument(
        "--est_stor_gw_from_rch",
        dest="estimateStorGroundwaterIniFromRecharge",
    )
    run_arguments.add_argument(
        "--day_gw_rch_ini",
        dest="dailyGroundwaterRechargeIni",
    )
    run_arguments.add_argument(
        "--qcf",
        dest="configuration_file_for_qualloc",
        help="QUAlloc configuration file",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.ini is None) == (args.ini_positional is None):
        parser.error("give the ini file once, positionally or with --ini")
    args.ini = os.path.abspath(args.ini or args.ini_positional)
    args.debug = args.debug or args.debug_word is not None
    return args


def collect_run_arguments(args):
    run_arguments = {}
    for action in build_parser().run_arguments._group_actions:
        value = getattr(args, action.dest)
        if value is None:
            continue
        run_arguments[action.dest] = value
        print(
            f"'{action.dest}' is set based on the system argument "
            f"({action.option_strings[0]}): {value}"
        )
    return run_arguments


def main():

    disclaimer.print_disclaimer()

    args = parse_args()
    run_arguments = collect_run_arguments(args)

    configuration = Configuration(
        iniFileName=args.ini,
        debug_mode=args.debug,
        system_arguments=sys.argv,
        run_arguments=run_arguments,
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
