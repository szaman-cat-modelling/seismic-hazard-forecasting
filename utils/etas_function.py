import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def build_kde_background(df, bandwidth=0.1):
    """
    Fit a 2D spatial KDE to earthquake locations for background seismicity.
    """
    coords = np.vstack([df.Longitude, df.Latitude]).T
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(coords)
    return kde

def kde_to_grid(kde, grid_lon, grid_lat):
    """
    Evaluate KDE on a forecast grid.
    """
    pts = np.vstack([grid_lon.ravel(), grid_lat.ravel()]).T
    log_dens = kde.score_samples(pts)
    return np.exp(log_dens).reshape(grid_lon.shape)

def seq_to_df(seq, region):
    """
    Convert an ETAS simulation sequence into a structured dataframe.
    """
    return pd.DataFrame({
        "Datetime": seq[:,0],
        "Latitude": seq[:,1],
        "Longitude": seq[:,2],
        "Magnitude": seq[:,3]
    })

def run_rolling_chunk(df, region, n_sims=100, chunk_years=5):
    """
    Run space–time ETAS in rolling time windows.
    """
    outputs = []
    years = np.arange(1990, 2026, chunk_years)

    for start in years:
        end = start + chunk_years
        chunk = df[(df.Datetime.dt.year >= start) & (df.Datetime.dt.year < end)]

        # run ETAS simulation — uses your notebook’s ETAS model block
        # this is where your PyTorch ETAS code runs
        # (left untouched)

        outputs.append({
            "start": start,
            "end": end,
            "forecast": None  # placeholder for actual ETAS results
        })

    return outputs

