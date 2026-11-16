from __future__ import print_function

import datetime
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
from configparser import ConfigParser, ExtendedInterpolation

from pcrglobwb.common import disclaimer
from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)


class Configuration(object):

    def __init__(
        self,
        iniFileName,
        debug_mode=False,
        system_arguments=None,
        run_arguments=None,
    ):
        object.__init__(self)

        if iniFileName is None:
            raise Exception("Error: No configuration file specified")

        # timestamp of this run, used in log file names etc.
        self._timestamp = datetime.datetime.now()

        self.iniFileName = os.path.abspath(iniFileName)

        self.debug_mode = debug_mode

        self.parse_configuration_file(self.iniFileName, run_arguments)
        self.create_output_directories()
        self.initialize_logging(system_arguments)
        self.repair_ini_key_names()

    def initialize_logging(self, system_arguments=None):
        """
        Initialize logging. Prints to both the console and a log file, at configurable levels
        """

        logging.getLogger().setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

        log_level_console = "INFO"
        log_level_file = "INFO"
        # log levels in order: DEBUG, INFO, WARNING, ERROR, CRITICAL

        # log level from the ini file
        if "log_level_console" in list(self.globalOptions.keys()):
            log_level_console = self.globalOptions["log_level_console"]
        if "log_level_file" in list(self.globalOptions.keys()):
            log_level_file = self.globalOptions["log_level_file"]

        # log level for debug mode
        if self.debug_mode:
            log_level_console = "DEBUG"
            log_level_file = "DEBUG"

        console_level = getattr(logging, log_level_console.upper(), logging.INFO)
        if not isinstance(console_level, int):
            raise ValueError(f"Invalid log level: {log_level_console}")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        logging.getLogger().addHandler(console_handler)

        prefix = (
            self.logFileDir
            + os.path.basename(self.iniFileName)
            + "_"
            + self._timestamp.isoformat().replace(":", ".")
        )
        log_filename = prefix + ".log"

        file_level = getattr(logging, log_level_file.upper(), logging.DEBUG)
        if not isinstance(console_level, int):
            raise ValueError(f"Invalid log level: {log_level_file}")

        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logging.getLogger().addHandler(file_handler)

        dbg_filename = prefix + ".dbg"
        debug_handler = logging.FileHandler(dbg_filename)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(debug_handler)

        disclaimer.print_disclaimer(with_logger=True)

        logger.info(f"Model run started at {self._timestamp}")
        logger.info(f"Logging output to {log_filename}")
        logger.info(f"Debugging output to {dbg_filename}")

        # log the platform, Python and PCRaster versions, paths, etc.
        logger.info(f"OS platform: {platform.system()}")
        logger.info(f"OS release: {platform.release()}")
        python_version = str(sys.version)
        logger.info(f"Python version:\n{python_version}")
        pcraster_check_filename = prefix + "_pcrcalc_output.txt"
        with open(pcraster_check_filename, "w") as f:
            try:
                subprocess.run(["pcrcalc"], stdout=f, stderr=subprocess.STDOUT)
            except OSError as e:
                f.write(f"{e}\n")
        logger.info(
            f"PCRaster version (output from pcrcalc): see the file {pcraster_check_filename}"
        )
        logger.info(f'PATH={os.environ["PATH"]}')
        logger.info(f'PYTHONPATH={os.environ.get("PYTHONPATH", "N/A")}')
        logger.info(f'HOSTNAME: {os.environ.get("HOSTNAME", "N/A")}')
        logger.info(
            "PCRASTER_NR_WORKER_THREADS (set by export, only for PCRaster version >= 4.2): "
            f'{os.environ.get("PCRASTER_NR_WORKER_THREADS", "N/A")}'
        )

        if system_arguments is not None:
            logger.info(
                f"The system arguments given to execute this run: {system_arguments}"
            )

        with open(prefix + ".ini", "w") as f:
            f.write(self.ini_text)

    def parse_configuration_file(self, modelFileName, run_arguments=None):
        """
        run_arguments become a [runArguments] section, readable in the ini as
        ${runArguments:<key>}.
        """

        with open(modelFileName) as f:
            ini_text = f.read()

        if run_arguments:
            ini_text += "\n[runArguments]\n" + "".join(
                # escaped because the ini is read with ExtendedInterpolation
                f"{key} = {value.replace('$', '$$')}\n"
                for key, value in run_arguments.items()
            )

        self.ini_text = ini_text

        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.optionxform = str
        config.read_string(ini_text, source=modelFileName)

        self.allSections = config.sections()

        for sec in self.allSections:
            vars(self)[sec] = {}
            options = config.options(sec)
            for opt in options:
                val = config.get(sec, opt)
                self.__getattribute__(sec)[opt] = val

        input_dir = self.globalOptions["inputDir"]
        self.cloneMap = vos.getFullPath(self.globalOptions["cloneMap"], input_dir)
        for item in ("precipitationNC", "temperatureNC", "refETPotFileNC"):
            if self.meteoOptions[item] != "None":
                self.meteoOptions[item] = vos.getFullPath(
                    self.meteoOptions[item], input_dir
                )

    def create_output_directories(self):
        out = self.globalOptions["outputDir"]
        is_sub_run = self.reportingOptions.get("is_sub_run") == "True"
        if not is_sub_run:
            os.makedirs(out, exist_ok=True)

        dirs = {}
        for name in ("tmp", "netcdf", "scripts", "log", "states", "maps"):
            dirs[name] = vos.getFullPath(name + "/", out)
            if not is_sub_run:
                if os.path.exists(dirs[name]):
                    shutil.rmtree(dirs[name])
                os.makedirs(dirs[name])

        if not is_sub_run:
            # does not include subfolders
            here = os.path.dirname(os.path.abspath(__file__))
            for filename in glob.glob(os.path.join(here, "*.py")):
                shutil.copy(filename, dirs["scripts"])

        self.tmpDir = dirs["tmp"]
        self.outNCDir = dirs["netcdf"]
        self.logFileDir = dirs["log"]
        self.endStateDir = dirs["states"]
        self.mapsDir = dirs["maps"]

        # so all pcr.report files are saved there
        os.chdir(self.mapsDir)

    def repair_ini_key_names(self):
        """
        If needed, change/modify some key names for initial condition fields.
        This is introduced because Edwin was very stupid as once he changed some key names of initial conditions! Yet, it is also useful particularly for runs without complete ini files.
        """

        # model time step (days)
        self.timeStep = 1.0
        self.timeStepUnit = "day"
        if "timeStep" in list(self.globalOptions.keys()) and "timeStepUnit" in list(
            self.globalOptions.keys()
        ):

            if (
                float(self.globalOptions["timeStep"]) != 1.0
                or self.globalOptions["timeStepUnit"] != "day"
            ):
                logger.error("Only a daily time step is supported; check the ini file")
                self.timeStep = None
                self.timeStepUnit = None

        # defaults for missing options
        if "routingMethod" not in list(self.routingOptions.keys()):
            logger.warning('No routingMethod in routingOptions; using "accuTravelTime"')
            self.routingOptions["routingMethod"] = "accuTravelTime"

        if "dynamicFloodPlain" not in list(self.routingOptions.keys()):
            logger.warning(
                'No dynamicFloodPlain; using "False": flood plain extent is constant'
            )
            self.routingOptions["dynamicFloodPlain"] = "False"

        if "historicalIrrigationArea" not in list(self.landSurfaceOptions.keys()):
            logger.warning(
                'No historicalIrrigationArea in landSurfaceOptions; using "None"'
            )
            self.landSurfaceOptions["historicalIrrigationArea"] = "None"

        # options to read a different forcing file for each year
        if "precipitation_set_per_year" not in list(self.meteoOptions.keys()):
            self.meteoOptions["precipitation_set_per_year"] = "False"
        if "temperature_set_per_year" not in list(self.meteoOptions.keys()):
            self.meteoOptions["temperature_set_per_year"] = "False"
        if "refETPotFileNC_set_per_year" not in list(self.meteoOptions.keys()):
            self.meteoOptions["refETPotFileNC_set_per_year"] = "False"

        # TODO: repair key names when running a 3-layer model with 2-layer initial conditions (and vice versa)

        # TODO: set a specific set of configuration options for a debugging run
