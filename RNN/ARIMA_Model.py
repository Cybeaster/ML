from pathlib import Path

import pandas as pd
import tensorflow as tf


path = Path("CTA_-_Ridership_-_Daily_Boarding_Totals.csv")

df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]  # shorter names
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()  # remove duplicated months (2011-10 and 2014-07)

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq(
    "D")  # take the rail column, select the dates between origin and today, and resample to daily frequency


model = ARIMA(rail_series, order=(1, 0, 0)  # p = 1, d = 0, q = 0
              , seasonal_order=(0, 1, 1, 7))  # P = 0, D = 1, Q = 1, S = 7
model = model.fit()

