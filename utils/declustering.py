import numpy as np
from haversine import haversine


def gardner_knopoff(df, time_window_func, dist_window_func):
    """
    Decluster using Gardner & Knopoff (1974) windows.
    df must have: Datetime, Latitude, Longitude, Magnitude
    """
    df = df.sort_values("Datetime").reset_index(drop=True)
    is_mainshock = np.ones(len(df), dtype=bool)

    for i, row in df.iterrows():
        if not is_mainshock[i]:
            continue  # skip if already flagged as aftershock

        mw = row["Magnitude"]
        t_win = time_window_func(mw)
        d_win = dist_window_func(mw)
        t0 = row["Datetime"]

        for j in range(i + 1, len(df)):
            dt = (df.loc[j, "Datetime"] - t0).days
            if dt > t_win:
                break  # stop if beyond window

            dist = haversine(
                (row["Latitude"], row["Longitude"]),
                (df.loc[j, "Latitude"], df.loc[j, "Longitude"])
            )

            if dist <= d_win:
                is_mainshock[j] = False

    return df[is_mainshock]


# Gardner–Knopoff (1974) window functions
def time_window(mw):
    return 10 ** (0.032 * mw + 2.738) / 1440  # days


def dist_window(mw):
    return 10 ** (0.1238 * mw + 0.983)  # km
