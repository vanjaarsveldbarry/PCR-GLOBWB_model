import argparse
import logging
import sys

from pcraster.framework import DynamicFramework, DynamicModel

from common import disclaimer
from common.currTimeStep import ModelTime
from pcrglobwb.configuration import INI_PLACEHOLDER_FLAGS, Configuration
from pcrglobwb.pcrglobwb import PCRGlobWB
from pcrglobwb.reporting import Reporting
from pcrglobwb.spinUp import SpinUp

logger = logging.getLogger(__name__)


class DeterministicRunner(DynamicModel):

    def __init__(self, configuration, modelTime, initialState=None):
        DynamicModel.__init__(self)

        self.modelTime = modelTime
        self.model = PCRGlobWB(configuration, modelTime, initialState)
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Run PCR-GLOBWB, filling the ini file's placeholders from options.",
        allow_abbrev=False,
    )
    parser.add_argument("ini_file", help="PCR-GLOBWB ini configuration file")
    parser.add_argument("--debug", action="store_true", help="log at DEBUG level")
    parser.add_argument(
        "--parallel",
        type=int,
        metavar="CLONE_CODE",
        help="run one clone of a parallel set, writing to the M<CLONE_CODE> subfolder",
    )
    for token, flag in INI_PLACEHOLDER_FLAGS.items():
        parser.add_argument(
            flag, dest=token, metavar="VALUE", help=f"value for {token} in the ini"
        )

    return parser.parse_args(argv)


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_arguments(argv)

    disclaimer.print_disclaimer()

    configuration = Configuration.from_arguments(args, system_arguments=argv)

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
                configuration, currTimeStep, initial_state
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
        configuration, currTimeStep, initial_state
    )

    dynamic_framework = DynamicFramework(
        deterministic_runner, currTimeStep.nrOfTimeSteps
    )
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()


if __name__ == "__main__":
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())
