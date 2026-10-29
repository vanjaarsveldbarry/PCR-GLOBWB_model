import calendar
import datetime
import logging
import sys

from common import utils

logger = logging.getLogger(__name__)


class model_time(object):
    # timer of the model: initializes and updates the continuous timer and
    # julian day number

    def __init__(
        self,
        startyear,
        endyear,
        time_increment,
        time_step_length=1.00,
    ):

        object.__init__(self)

        self.seconds_per_day = 86400

        self.time_step_length = time_step_length
        self.startyear = startyear
        self.endyear = endyear
        self.time_increment = time_increment
        self.startdate = datetime.datetime(self.startyear, 1, 1)
        self.enddate = datetime.datetime(self.endyear, 12, 31)
        self.date = self.startdate

        if self.time_increment != "daily":
            sys.exit(f"{self.time_increment} cannot be used")

        # length of the simulation
        self.number_time_steps = (
            int(((self.enddate - self.date).days) / self.time_step_length) + 1
        )

        self.report_flags = {}.fromkeys(["daily", "monthly", "yearly"], False)

        # special markers, all False by default
        self.last_day_of_month = False
        self.last_day_of_year = False
        self.last_time_step = False

    def __repr__(self):
        return (
            "this is an instance of the Caleros time class object with date %s"
            % self.__str__
        )

    def __str__(self):
        return self.date

    def update(self, increment):
        # increment the date by one time step
        self.time_step_length = 1
        self.date = self.startdate + datetime.timedelta(
            days=(increment - 1) * self.time_step_length
        )

        self.year = self.date.year
        month = self.date.month
        day = self.date.day

        self.julianday = utils.get_julian_day_number(self.date)

        # special markers
        self.report_flags["daily"] = True
        number_days = calendar.monthrange(self.year, month)[1]
        self.report_flags["monthly"] = day == number_days
        if self.report_flags["monthly"] and month == 12:
            self.last_day_of_year = True
        else:
            self.last_day_of_year = False
        self.report_flags["yearly"] = self.last_day_of_year

        self.last_time_step = increment == self.number_time_steps

        logger.info(f"processing {self.date}")
