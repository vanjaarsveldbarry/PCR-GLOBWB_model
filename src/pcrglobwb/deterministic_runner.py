import argparse
import logging
import os
import sys

from pcraster.framework import DynamicFramework, DynamicModel

from common import disclaimer
from common.currTimeStep import ModelTime
from pcrglobwb.configuration import Configuration
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

    def initial(self):
        pass

    def dynamic(self):

        # update the model time from the current PCRaster time step
        self.modelTime.update(self.currentTimeStep())

        # update the model (uses the current model time)

        self.model.read_forcings()
        self.model.update(report_water_balance=True)

        self.reporting.report()


def run_framework(runner):
    framework = DynamicFramework(runner, runner.modelTime.nrOfTimeSteps)
    framework.setQuiet(True)
    framework.run()


def spin_up(configuration, model_time):
    max_cycles = int(configuration.globalOptions["maxSpinUpsInYears"])
    if max_cycles <= 0:
        return None

    logger.info(f"Spin-Up #Total Years: {max_cycles}")

    convergence = SpinUp(configuration)
    state = None

    for cycle in range(1, max_cycles + 1):
        model_time.setSpinUpCycle(
            configuration.globalOptions["startTime"], cycle, max_cycles
        )
        logger.info(f"Spin-Up Run No. {cycle}")
        runner = DeterministicRunner(configuration, model_time, state)

        # getAllState adds the grid totals the convergence check needs;
        # getState is what the next run restarts from
        all_state_begin = runner.model.getAllState()
        run_framework(runner)
        all_state_end = runner.model.getAllState()

        has_converged = convergence.checkConvergence(
            all_state_begin, all_state_end, cycle, runner.model.routing.cellArea
        )
        convergence.archive_netcdf_output()

        state = runner.model.getState()

        if has_converged:
            return state

    logger.warning(f"Spin-up did not converge after {max_cycles} cycles.")
    return state


def run(configuration):
    model_time = ModelTime()

    initial_state = spin_up(configuration, model_time)

    model_time.setTransientPeriod(
        configuration.globalOptions["startTime"], configuration.globalOptions["endTime"]
    )
    logger.info("Transient simulation run started.")
    runner = DeterministicRunner(configuration, model_time, initial_state)
    run_framework(runner)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Run PCR-GLOBWB from an ini file.", allow_abbrev=False
    )
    parser.add_argument("ini_file", help="PCR-GLOBWB ini configuration file")
    parser.add_argument("--debug", action="store_true", help="log at DEBUG level")
    return parser.parse_args(argv)


def main(argv=None):

    args = parse_arguments(argv)

    disclaimer.print_disclaimer()

    configuration = Configuration(
        iniFileName=os.path.abspath(args.ini_file),
        replacements={},
        debug_mode=args.debug,
    )
    configuration.set_configuration()

    run(configuration)


if __name__ == "__main__":
    sys.exit(main())
