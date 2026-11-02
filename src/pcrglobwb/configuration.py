from __future__ import print_function

import datetime
import glob
import logging
import os
import platform
import re
import shutil
import sys
from configparser import RawConfigParser as ConfigParser

from common import disclaimer, utils

logger = logging.getLogger(__name__)

# ini placeholder -> command-line flag that supplies its value
INI_PLACEHOLDER_FLAGS = {
    "MAIN_OUTPUT_DIR": "--mod",
    "MAIN_INPUT_DIR": "--mid",
    "START_DATE": "--sd",
    "END_DATE": "--ed",
    "MAIN_INITIAL_STATE_FOLDER": "--misd",
    "DATE_FOR_INITIAL_STATES": "--dfis",
    "CLONE_CONNECTIONS": "--clone_connections",
    "PRECIPITATION_FORCING_FILE": "--pff",
    "TEMPERATURE_FORCING_FILE": "--tff",
    "REF_POT_ET_FORCING_FILE": "--rpetff",
    "PRESSURE_FORCING_FILE": "--presff",
    "WIND_FORCING_FILE": "--windff",
    "SHORTWAVE_RADIATION_FORCING_FILE": "--swradff",
    "RELATIVE_HUMIDITY_FORCING_FILE": "--relhumff",
    "BASEFLOW_EXP_INPUT": "--bfexp",
    "NUMBER_OF_SPINUP_YEARS": "--num_of_sp_years",
    "CLONEMAP": "--clonemap",
    "USE_MAXIMUM_STOR_GROUNDWATER_FOSSIL_INI": "--use_max_fossil_gw_ini",
    "ESTIMATE_STOR_GROUNDWATER_INI_FROM_RECHARGE": "--est_stor_gw_from_rch",
    "DAILY_GROUNDWATER_RECHARGE_INI": "--day_gw_rch_ini",
    "QUALLOC_CONFIG_FILE": "--qcf",
    "MIN_SOIL_DEPTH_FRAC_MULTIPLIER": "--min_soil_depth_frac_mult",
    "KSAT_LOG10_MULTIPLIER": "--ksat_log10_mult",
    "RECESSION_COEFF_LOG10_MULTIPLIER": "--recession_coeff_log10_mult",
    "STOR_CAP_MULTIPLIER": "--stor_cap_mult",
    "DEGREE_DAY_FACTOR_MULTIPLIER": "--degree_day_factor_mult",
    "REF_POT_ET_MULTIPLIER": "--ref_pot_et_mult",
    "MANNINGS_N_MULTIPLIER": "--mannings_n_mult",
    "STOR_GROUNDWATER_INI_FILE": "--stor_groundwater_ini_file",
}

placeholder_pattern = re.compile(rf"\b({'|'.join(INI_PLACEHOLDER_FLAGS)})\b")


class Configuration(object):

    def __init__(
        self,
        iniFileName,
        replacements,
        debug_mode=False,
        relative_ini_meteo_paths=False,
    ):
        object.__init__(self)

        if iniFileName is None:
            raise Exception("Error: No configuration file specified")

        # timestamp of this run, used in log file names etc.
        self._timestamp = datetime.datetime.now()

        self.iniFileName = os.path.abspath(iniFileName)

        self.debug_mode = debug_mode

        # save the cwd; it may be changed later by some utility functions
        self._cwd = os.getcwd()

        self.replacements = replacements
        with open(self.iniFileName) as ini_file:
            ini_template = ini_file.read()

        self.check_placeholders(ini_template)
        self.iniContent = placeholder_pattern.sub(
            lambda match: self.replacements.get(match[1], match[1]),
            ini_template,
        )

        self.parse_configuration_file(self.iniContent)

        # option to run in a sandbox with meteo files and initial conditions
        self.using_relative_path_for_output_directory = False
        # reported in initialize_logging, as the logger does not exist yet
        self.missing_ini_files = []
        if relative_ini_meteo_paths:
            self.using_relative_path_for_output_directory = True
            self.make_ini_meteo_paths_absolute()

    @classmethod
    def from_arguments(cls, args, system_arguments=None):
        replacements = {
            token: getattr(args, token)
            for token in INI_PLACEHOLDER_FLAGS
            if getattr(args, token) is not None
        }

        configuration = cls(
            iniFileName=args.ini_file,
            replacements=replacements,
            debug_mode=args.debug,
        )

        # a parallel run (e.g. 5 and 6 arcmin runs) gets a specific directory based on the clone code:
        if args.parallel is not None:
            options = configuration.globalOptions
            options["outputDir"] += f"/M{args.parallel:02}"
            options["cloneMap"] = options["cloneMap"] % args.parallel
            # landmask for the model calculation
            if options["landmask"] != "None":
                options["landmask"] = options["landmask"] % args.parallel

        configuration.set_configuration(system_arguments=system_arguments)

        return configuration

    def set_configuration(self, system_arguments=None):

        self.set_input_files()
        self.create_output_directories()

        self.initialize_logging("Default", system_arguments)

        self.backup_configuration()

        self.repair_ini_key_names()

    def make_ini_meteo_paths_absolute(self):
        for section in self.allSections:
            sec = getattr(self, section)

            for key, value in list(sec.items()):
                if key.endswith("Ini") and value is not None and value != "None":
                    sec[key] = os.path.abspath(value)

                if key == "precipitationNC" or key == "temperatureNC":
                    sec[key] = os.path.abspath(value)

            for key, value in list(sec.items()):
                if key.endswith("Ini") and value is not None and value != "None":
                    if not os.path.exists(value):
                        self.missing_ini_files.append((key, value))

    # make paths absolute to the cwd at the time the configuration was created
    def make_absolute_path(self, path):
        return os.path.normpath(os.path.join(self._cwd, path))

    def initialize_logging(self, log_file_location="Default", system_arguments=None):
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
            raise ValueError("Invalid log level: %s", log_level_console)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        logging.getLogger().addHandler(console_handler)

        if log_file_location != "Default":
            self.logFileDir = log_file_location
        log_filename = (
            self.logFileDir
            + os.path.basename(self.iniFileName)
            + "_"
            + str(self._timestamp.isoformat()).replace(":", ".")
            + ".log"
        )

        file_level = getattr(logging, log_level_file.upper(), logging.DEBUG)
        if not isinstance(console_level, int):
            raise ValueError("Invalid log level: %s", log_level_file)

        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logging.getLogger().addHandler(file_handler)

        # debug log file name
        dbg_filename = (
            self.logFileDir
            + os.path.basename(self.iniFileName)
            + "_"
            + str(self._timestamp.isoformat()).replace(":", ".")
            + ".dbg"
        )

        debug_handler = logging.FileHandler(dbg_filename)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(debug_handler)

        disclaimer.print_disclaimer(with_logger=True)

        logger.info("Model run started at %s", self._timestamp)
        logger.info("Logging output to %s", log_filename)
        logger.info("Debugging output to %s", dbg_filename)
        for key, value in self.missing_ini_files:
            logger.warning("%s: initial condition file %s does not exist", key, value)

        # log the platform, Python and PCRaster versions, paths, etc.
        logger.info("OS platform: %s", str(platform.system()))
        logger.info("OS relesase: %s", str(platform.release()))
        python_version = str(sys.version)
        logger.info("Python version:\n%s", python_version)
        pcraster_check_filename = (
            self.logFileDir
            + os.path.basename(self.iniFileName)
            + "_"
            + str(self._timestamp.isoformat()).replace(":", ".")
            + "_pcrcalc_output.txt"
        )
        cmd = f"pcrcalc &> {pcraster_check_filename}"
        os.system(cmd)
        logger.info(
            "PCRaster version (output from pcrcalc): see the file %s",
            pcraster_check_filename,
        )
        logger.info("PATH=%s", os.environ["PATH"])
        logger.info("PYTHONPATH=%s", os.environ.get("PYTHONPATH", "N/A"))
        logger.info("HOSTNAME: %s", os.environ.get("HOSTNAME", "N/A"))
        logger.info(
            "PCRASTER_NR_WORKER_THREADS (set by export, only for PCRaster version >= 4.2): %s",
            os.environ.get("PCRASTER_NR_WORKER_THREADS", "N/A"),
        )

        if system_arguments is not None:
            logger.info(
                "The system arguments given to execute this run: %s", system_arguments
            )
        for token, value in self.replacements.items():
            logger.info("Ini placeholder %s is set to %s", token, value)

    def backup_configuration(self):

        backup_filename = (
            self.logFileDir
            + os.path.basename(self.iniFileName)
            + "_"
            + str(self._timestamp.isoformat()).replace(":", ".")
            + ".ini"
        )
        with open(backup_filename, "w") as backup_file:
            backup_file.write(self.iniContent)

    def parse_configuration_file(self, ini_content):

        config = ConfigParser()
        config.optionxform = str
        config.read_string(ini_content)

        self.allSections = config.sections()

        for sec in self.allSections:
            vars(self)[sec] = {}
            options = config.options(sec)
            for opt in options:
                val = config.get(sec, opt)
                self.__getattribute__(sec)[opt] = val

    def check_placeholders(self, ini_template):
        config = ConfigParser()
        config.optionxform = str
        config.read_string(ini_template)

        # parsed values, not the raw text, so commented-out ini lines don't count
        in_ini = {
            token
            for section in config.sections()
            for value in config[section].values()
            for token in placeholder_pattern.findall(value)
        }

        problems = {
            "placeholders without a value": in_ini - self.replacements.keys(),
            "values for placeholders it does not contain": self.replacements.keys()
            - in_ini,
        }
        messages = [
            f"{problem}: "
            + ", ".join(f"{t} ({INI_PLACEHOLDER_FLAGS[t]})" for t in sorted(tokens))
            for problem, tokens in problems.items()
            if tokens
        ]
        if messages:
            sys.exit(f"{self.iniFileName} has {'; '.join(messages)}")

    def set_input_files(self):
        self.cloneMap = utils.getFullPath(
            self.globalOptions["cloneMap"], self.globalOptions["inputDir"]
        )

        # full paths of the input directories/files
        dirsAndFiles = ["precipitationNC", "temperatureNC", "refETPotFileNC"]
        for item in dirsAndFiles:
            if self.meteoOptions[item] != "None":
                self.meteoOptions[item] = utils.getFullPath(
                    self.meteoOptions[item], self.globalOptions["inputDir"]
                )

    def create_output_directories(self):

        if (
            "is_sub_run" in list(self.reportingOptions.keys())
            and self.reportingOptions["is_sub_run"] == "True"
        ):
            if self.using_relative_path_for_output_directory:
                self.globalOptions["outputDir"] = self.make_absolute_path(
                    self.globalOptions["outputDir"]
                )

            self.tmpDir = utils.getFullPath("tmp/", self.globalOptions["outputDir"])

            self.outNCDir = utils.getFullPath(
                "netcdf/", self.globalOptions["outputDir"]
            )

            # backup of the Python scripts used
            self.scriptDir = utils.getFullPath(
                "scripts/", self.globalOptions["outputDir"]
            )

            # starting directory where all scripts are stored
            path_of_this_module = os.path.abspath(os.path.dirname(__file__))
            self.starting_directory = path_of_this_module

            self.logFileDir = utils.getFullPath("log/", self.globalOptions["outputDir"])

            self.endStateDir = utils.getFullPath(
                "states/", self.globalOptions["outputDir"]
            )

            self.mapsDir = utils.getFullPath("maps/", self.globalOptions["outputDir"])

            # go to the PCRaster maps directory (so all pcr.report files are saved there)
            os.chdir(self.mapsDir)

        else:
            if self.using_relative_path_for_output_directory:
                self.globalOptions["outputDir"] = self.make_absolute_path(
                    self.globalOptions["outputDir"]
                )

            # root/parent of the output directory
            cleanOutputDir = False
            if cleanOutputDir:
                try:
                    shutil.rmtree(self.globalOptions["outputDir"])
                except Exception:
                    # new outputDir (does not exist yet)
                    pass
            try:
                os.makedirs(self.globalOptions["outputDir"])
            except Exception:
                # new outputDir (does not exist yet)
                pass

            self.tmpDir = utils.getFullPath("tmp/", self.globalOptions["outputDir"])

            if os.path.exists(self.tmpDir):
                shutil.rmtree(self.tmpDir)
            os.makedirs(self.tmpDir)

            self.outNCDir = utils.getFullPath(
                "netcdf/", self.globalOptions["outputDir"]
            )
            if os.path.exists(self.outNCDir):
                shutil.rmtree(self.outNCDir)
            os.makedirs(self.outNCDir)

            # backup of the Python scripts used
            self.scriptDir = utils.getFullPath(
                "scripts/", self.globalOptions["outputDir"]
            )

            if os.path.exists(self.scriptDir):
                shutil.rmtree(self.scriptDir)
            os.makedirs(self.scriptDir)

            # starting directory where all scripts are stored
            path_of_this_module = os.path.abspath(os.path.dirname(__file__))
            self.starting_directory = path_of_this_module

            for filename in glob.glob(os.path.join(path_of_this_module, "*.py")):
                shutil.copy(filename, self.scriptDir)
            # TODO: fix this copying (it does not include subfolders)

            self.logFileDir = utils.getFullPath("log/", self.globalOptions["outputDir"])
            cleanLogDir = True
            if os.path.exists(self.logFileDir) and cleanLogDir:
                shutil.rmtree(self.logFileDir)
            os.makedirs(self.logFileDir)

            self.endStateDir = utils.getFullPath(
                "states/", self.globalOptions["outputDir"]
            )
            if os.path.exists(self.endStateDir):
                shutil.rmtree(self.endStateDir)
            os.makedirs(self.endStateDir)

            self.mapsDir = utils.getFullPath("maps/", self.globalOptions["outputDir"])
            cleanMapDir = True
            if os.path.exists(self.mapsDir) and cleanMapDir:
                shutil.rmtree(self.mapsDir)
            os.makedirs(self.mapsDir)

            # go to the PCRaster maps directory (so all pcr.report files are saved there)
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
