import argparse
import logging
import sys

from common import disclaimer
from pcrglobwb.configuration import INI_PLACEHOLDER_FLAGS, Configuration
from pcrglobwb.deterministic_runner import run

logger = logging.getLogger(__name__)


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

    args = parse_arguments(argv)

    disclaimer.print_disclaimer()

    configuration = Configuration.from_arguments(args)
    logger.info("The system arguments given to execute this run: %s", sys.argv[1:])

    run(configuration)


if __name__ == "__main__":
    sys.exit(main())
