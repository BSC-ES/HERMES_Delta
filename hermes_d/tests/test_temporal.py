import unittest
import shutil
import pandas as pd
from calendar import monthrange
from datetime import datetime
import os

from hermes_d.modules.temporal.temporal import TemporalDelta, calculate_date_array
from hermes_d.config import configure_logger

configure_logger(log_path=str("logs_unittests/log_temporal_testing.log"))


# noinspection DuplicatedCode
class TestTemporalDelta(unittest.TestCase):
    def setUp(self):
        self.temporal_delta = TemporalDelta(
            None, None, None, None, False, daily_consistency=True
        )

    def test_flat_profile(self):
        """
        Test that a flat weekly profile is correctly rebalanced and normalized.

        A uniform profile with equal weights (1) for each weekday is passed, along with January 2023 (31 days).
        After rebalancing, the profile should still be flat (same value across all days), and the sum of the
        profile weighted by weekday counts should equal 1.

        Input:
            - weekly_profile = [1, 1, 1, 1, 1, 1, 1] for Monday to Sunday
            - year = 2023, month = 1 (January)

        Expected behavior:
            - All output values should be equal
            - Weighted sum with weekday counts should be 1
        """
        weekly_profile = pd.Series(
            [1, 1, 1, 1, 1, 1, 1],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2023
        month = 1  # January (31 days)
        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York"],
            index=["P_month", "P_week", "P_hour", "tzid"],
        )
        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile, year, month, profile_info, holidays=False
        )
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        weekday_counts = days_in_month.weekday.value_counts().sort_index()

        # Check that the sum of (rebalanced_profile * weekday_counts) equals 1
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1, places=5)

        # Check that the profile remains flat after rebalancing
        for value in result:
            self.assertAlmostEqual(value, result.iloc[0])

    def test_leap_year_february(self):
        """
        Test that a weekly profile in February of a leap year is correctly rebalanced.

        A uniform profile with equal weights (1) for each weekday is passed, along with February 2024 (29 days).
        The test checks if the extra day of February is handled correctly, ensuring the weighted sum equals 1.

        Input:
            - weekly_profile = [1, 1, 1, 1, 1, 1, 1] for Monday to Sunday
            - year = 2024, month = 2 (February)

        Expected behavior:
            - The sum of the rebalanced profile weighted by weekday counts should be 1
        """
        weekly_profile = pd.Series(
            [1, 1, 1, 1, 1, 1, 1],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2024  # Leap year
        month = 2  # February

        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York"],
            index=["P_month", "P_week", "P_hour", "tzid"],
        )

        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile, year, month, profile_info, holidays=False
        )
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        weekday_counts = days_in_month.weekday.value_counts().sort_index()

        # Check that the sum of the rebalanced profile, weighted by the weekday counts, is 1
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1, places=5)

    def test_rebalance_weekly_profile_with_mocked_holidays(self):
        """
        Test that a weekly profile is correctly rebalanced while considering holidays.

        A weekly profile with reduced weights on Saturday and Sunday is passed, along with January 2023.
        The test mocks a holiday on January 6th, impacting the rebalancing. The output is validated to ensure
        the sum equals 1 and that Sunday has a higher or equal weight than Saturday.

        Input:
            - weekly_profile = [1, 1, 1, 1, 1, 0.5, 0.5] for Monday to Sunday
            - year = 2023, month = 1

        Expected behavior:
            - The sum of the rebalanced profile weighted by weekday counts should be 1
            - The weight on Sunday should be greater than or equal to Saturday
        """
        weekly_profile = pd.Series(
            [1, 1, 1, 1, 1, 0.5, 0.5],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2023
        month = 1

        from hermes_d.modules.temporal import temporal
        from holidays import HolidayBase

        class MockHolidays(HolidayBase):
            def __contains__(self, date):
                return date == datetime(2023, 1, 6).date()

        temporal.country_holidays = lambda *args, **kwargs: MockHolidays()
        self.temporal_delta.holiday_regions = {"ES51": "Catalonia"}

        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York", "ES51"],
            index=["P_month", "P_week", "P_hour", "tzid", "NUTS2_code"],
        )

        # result = self.temporal_delta.__rebalance_weekly_profile_holidays__(weekly_profile, year, month, 'ES51')
        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile,
            year,
            month,
            profile_info,
            holidays=True,
            holiday_regions=self.temporal_delta.holiday_regions,
        )

        # Recompute weekday counts using the same logic as the function
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        holidays = [day for day in days_in_month if day in MockHolidays()]
        weekday_counts = (
            days_in_month.to_series()
            .apply(lambda day: 6 if day in holidays else day.weekday())
            .value_counts()
            .sort_index()
        )

        # Validate rebalanced profile sum and check Sunday was impacted
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1.0, places=5)
        self.assertGreaterEqual(result[6], result[5])

    def test_proportional_profile(self):
        """
        Test that a proportional weekly profile is correctly rebalanced.

        A profile where each day has the same proportional weight (0.5) is passed, along with January 2023.
        The expected output is a rebalanced profile where the sum weighted by weekday counts equals 1.

        Input:
            - weekly_profile = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] for Monday to Sunday
            - year = 2023, month = 1

        Expected behavior:
            - The sum of the rebalanced profile weighted by weekday counts should be 1
        """
        weekly_profile = pd.Series(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2023
        month = 1

        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York"],
            index=["P_month", "P_week", "P_hour", "tzid"],
        )

        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile, year, month, profile_info, holidays=False
        )
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        weekday_counts = days_in_month.weekday.value_counts().sort_index()

        # Check that the sum of the rebalanced profile, weighted by the weekday counts, is 1
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1, places=5)

    def test_non_flat_profile(self):
        """
        Test that a non-flat weekly profile is correctly rebalanced.

        A profile with varying weights for each day of the week is passed, along with January 2024 (31 days).
        The expected output is a rebalanced profile where the sum weighted by weekday counts equals 1.

        Input:
            - weekly_profile = [1.02, 1.06, 1.08, 1.1, 1.14, 0.81, 0.79] for Monday to Sunday
            - year = 2024, month = 1

        Expected behavior:
            - The sum of the rebalanced profile weighted by weekday counts should be 1
        """
        weekly_profile = pd.Series(
            [1.02, 1.06, 1.08, 1.1, 1.14, 0.81, 0.79],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2024
        month = 1  # Jan (31 days)

        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York"],
            index=["P_month", "P_week", "P_hour", "tzid"],
        )

        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile, year, month, profile_info, holidays=False
        )
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        weekday_counts = days_in_month.weekday.value_counts().sort_index()

        # Check that the sum of the rebalanced profile, weighted by the weekday counts, is 1
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1, places=5)

    def test_incomplete_profile_weekday(self):
        """
        Test error handling for an incomplete weekly profile.

        An incomplete profile string is passed that lacks the 'weekend' key. The test checks if a ValueError
        is raised as expected due to the missing key.

        Input:
            - incomplete_profile = "weekday=H_weekday" (missing 'weekend')

        Expected behavior:
            - A ValueError should be raised indicating the missing key.
        """
        incomplete_profile = "weekday=H_weekday"  # Missing 'weekend' key
        with self.assertRaises(ValueError):
            self.temporal_delta.__parse_profile_string__(incomplete_profile)

    def test_holidays_profile(self):
        """
        Test that a profile is correctly rebalanced while considering holidays.

        A non-flat weekly profile is passed, along with May 2024. The test checks if holidays are treated
        as Sundays during rebalancing, ensuring the sum weighted by weekday counts equals 1.

        Input:
            - weekly_profile = [1.02, 1.06, 1.08, 1.1, 1.14, 0.81, 0.79] for Monday to Sunday
            - year = 2024, month = 5 (May)

        Expected behavior:
            - The sum of the rebalanced profile weighted by weekday counts should be 1
        """
        weekly_profile = pd.Series(
            [1.02, 1.06, 1.08, 1.1, 1.14, 0.81, 0.79],
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        year = 2024
        month = 5  # May (example month)

        profile_info = pd.Series(
            ["M_0001", "W_0001", {"default": "H_0001"}, "America/New_York", "US"],
            index=["P_month", "P_week", "P_hour", "tzid", "NUTS2_code"],
        )

        # Mock the holiday regions to ensure holidays are treated as Sundays
        self.temporal_delta.holiday_regions = {"US": "NY"}

        from holidays import country_holidays

        country_code = "US"
        holidays = country_holidays(
            country_code, subdiv=self.temporal_delta.holiday_regions["US"], years=[year]
        )
        num_days_in_month = monthrange(year, month)[1]
        days_in_month = pd.date_range(
            start=f"{year}-{month}-01", periods=num_days_in_month
        )
        weekday_counts = (
            days_in_month.to_series()
            .apply(lambda day: 6 if day in holidays else day.weekday())
            .value_counts()
            .sort_index()
        )

        result = self.temporal_delta.__rebalance_weekly_profile__(
            weekly_profile,
            year,
            month,
            profile_info,
            holidays=True,
            holiday_regions=self.temporal_delta.holiday_regions,
        )

        # Check the result is correctly rebalanced
        total_sum = (result * weekday_counts).sum()
        self.assertAlmostEqual(total_sum, 1, places=3)

    def test_different_timezones(self):
        """
        Test conversion of profiles across different timezones.

        The test ensures that monthly, weekly, and hourly profiles are correctly adjusted for timezone differences.
        The input profiles are structured with a mapping to the respective time periods.

        Input:
            - profiles = {
                'monthly': pd.DataFrame([[1] * 12], index=['M_0001']),
                'weekly': pd.DataFrame([[1] * 7], index=['W_0001']),
                'hourly': pd.DataFrame([[1] * 24], index=['H_0001']),
                'daily': pd.DataFrame([[1] * 31], index=['D_0001'])
              }
            - temp_mapping = {
                'P_month': ['M_0001'],
                'P_week': ['W_0001'],
                'P_hour': ["{'default': 'H_0001'}"],
                'P_day': ['D_0001'],
                'tzid': ['America/New_York']
              }

        Expected behavior:
            - The resulting shape of the profiles should match the number of time steps
        """

        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        profiles = {
            "monthly": pd.DataFrame([[1] * 12], index=["M_0001"], columns=months),
            "weekly": pd.DataFrame([[1] * 7], index=["W_0001"], columns=weekdays),
            "hourly": pd.DataFrame(
                [[1] * 24],
                index=["H_0001"],
                columns=[str(i) for i in list(range(1, 25))],
            ),
        }

        temp_mapping = pd.DataFrame(
            {
                "P_month": ["M_0001"],
                "P_week": ["W_0001"],
                "P_hour": ["{'default': 'H_0001'}"],
                "P_day": ["D_0001"],
                "tzid": ["America/New_York"],
            },
            index=["T_0001"],
        )

        time_steps = pd.date_range(start="2023-01-01", periods=24, freq="h", tz="UTC")
        # temporal_delta = TemporalDelta(None, None, {}, time_steps, False, daily_consistency=True)
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        result = self.temporal_delta.__calculate_yr_to_hr_profiles__(
            profiles, temp_mapping, daily=False
        )
        # Verifies timezone conversion and resulting factors
        self.assertEqual(result.shape[1], len(time_steps))

    def test_calculate_date_array_invalid_params(self):
        """
        Test calculate_date_array with invalid parameters.

        The test checks if ValueError is raised for invalid timestep_num and timestep_freq inputs.

        Input:
            - Invalid inputs for timestep_num = 0 and timestep_freq = 1
            - Invalid inputs for timestep_num = 10 and timestep_freq = 0

        Expected behavior:
            - A ValueError should be raised for each invalid input
        """
        with self.assertRaises(ValueError):
            calculate_date_array(datetime(2023, 1, 1), "hourly", 0, 1)
        with self.assertRaises(ValueError):
            calculate_date_array(datetime(2023, 1, 1), "hourly", 10, 0)

    def test_calculate_date_array_intervals(self):
        """
        Test calculate_date_array for generating date series at specified intervals.

        The test checks the output for hourly and daily intervals, ensuring the correct datetime objects are returned.

        Input:
            - start_date = datetime(2023, 1, 1)
            - timestep_num = 3, timestep_freq = 1 for both hourly and daily intervals

        Expected behavior:
            - For hourly intervals, the output should be:
                - [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0), datetime(2023, 1, 1, 2, 0)]
            - For daily intervals, the output should be:
                - [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        """
        start_date = datetime(2023, 1, 1)
        result_hourly = calculate_date_array(start_date, "hourly", 3, 1)
        self.assertEqual(
            result_hourly,
            [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 1, 0),
                datetime(2023, 1, 1, 2, 0),
            ],
        )

        result_daily = calculate_date_array(start_date, "daily", 3, 1)
        self.assertEqual(
            result_daily,
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        )

    def test_empty_seie_both(self):
        """
        Test that an empty SEIE file raises an error in the initialization.
        """

        with self.assertRaises(ValueError):
            TemporalDelta(
                pd.DataFrame({}), None, None, None, False, daily_consistency=True
            )

    def test_leapyear_profiles_nonleapyear_year(self):
        # for the case where daily profiles are based on leap years, but the year
        # of the simulation is a non-leap year.

        # assert that an error is raised.

        time_steps = pd.date_range(
            start="2023-06-15 11:00:00", end="2023-06-15 14:00:00", freq="h", tz="UTC"
        )  # 365-day year
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        df_unique = pd.DataFrame(
            [[1] * 366], columns=list(range(1, 367)), index=["D_0001"]
        )
        id_mapping = {"D_020103_ES12": "D_0001"}

        with self.assertRaises(ValueError):
            self.temporal_delta._check_temporal_mapping(
                df_unique, id_mapping, temp_res="daily"
            )

    def test_leapyear_profiles_acrossyears(self):
        # for the case where daily profiles are based on leap years, but one of the years
        # of the simulation is a non-leap year.

        # assert that an error is raised.

        time_steps = pd.date_range(
            start="2023-12-31 22:00:00", end="2024-01-01 02:00:00", freq="h", tz="UTC"
        )  # 365-day year
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        df_unique = pd.DataFrame(
            [[1] * 366], columns=list(range(1, 367)), index=["D_0001"]
        )
        id_mapping = {"D_020103_ES12": "D_0001"}

        with self.assertRaises(ValueError):
            self.temporal_delta._check_temporal_mapping(
                df_unique, id_mapping, temp_res="daily"
            )

    def test_nonleapyear_profiles_leapyear_year(self):
        # for the case where daily profiles are based on leap years, but the year
        # of the simulation is a non-leap year.

        # assert that an error is raised.

        time_steps = pd.date_range(
            start="2024-06-15 11:00:00", end="2024-06-15 14:00:00", freq="h", tz="UTC"
        )  # 365-day year
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        df_unique = pd.DataFrame(
            [[1] * 365], columns=list(range(1, 366)), index=["D_0001"]
        )
        id_mapping = {"D_020103_ES12": "D_0001"}

        with self.assertRaises(ValueError):
            self.temporal_delta._check_temporal_mapping(
                df_unique, id_mapping, temp_res="daily"
            )

    def test_nonleapyear_profiles_acrossyears(self):
        # for the case where daily profiles are based on non-leap years, but one of the years
        # of the simulation is a leap year.

        # assert that an error is raised.

        time_steps = pd.date_range(
            start="2023-12-31 22:00:00", end="2024-01-01 02:00:00", freq="h", tz="UTC"
        )  # 365-day year
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        df_unique = pd.DataFrame(
            [[1] * 366], columns=list(range(1, 367)), index=["D_0001"]
        )
        id_mapping = {"D_020103_ES12": "D_0001"}

        with self.assertRaises(ValueError):
            self.temporal_delta._check_temporal_mapping(
                df_unique, id_mapping, temp_res="daily"
            )

    def test_column_sum_mismatch(self):
        # 5x73 =365, but the profiles only have 73 cols. should raise an error

        time_steps = pd.date_range(
            start="2023-06-15 11:00:00", end="2023-06-15 14:00:00", freq="h", tz="UTC"
        )  # 365-day year
        self.temporal_delta.time_step_list = time_steps
        self.temporal_delta.set_inconsistent_time_steps(time_steps)

        df_unique = pd.DataFrame(
            [[5] * 73], columns=list(range(1, 74)), index=["D_0001"]
        )
        id_mapping = {"D_020103_ES12": "D_0001"}

        with self.assertRaises(ValueError):
            self.temporal_delta._check_temporal_mapping(
                df_unique, id_mapping, temp_res="daily"
            )

    @classmethod
    def tearDownClass(cls):
        logfolder = "logs_unittests"
        if os.path.exists(logfolder):
            shutil.rmtree(logfolder)


if __name__ == "__main__":
    unittest.main()
