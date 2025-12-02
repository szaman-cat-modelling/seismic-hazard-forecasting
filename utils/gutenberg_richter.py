import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


##############################################################
# AEReLU magnitude transform
##############################################################

def AEReLU_torch(M, Mc, beta):
    """
    Computes effective magnitude using AEReLU function.
    M : array or tensor of magnitudes
    Mc : completeness magnitude
    beta : sharpness for AEReLU
    """
    if isinstance(M, np.ndarray):
        M_tensor = torch.from_numpy(M).float()
    else:
        M_tensor = M.float()

    softplus = nn.Softplus()
    x = softplus(beta * (M_tensor - Mc))
    M_eff = Mc + (1.0 / beta) * x
    return M_eff


##############################################################
# Gutenberg–Richter relation
##############################################################

def gr_equation(M, a, b, Mc, beta, use_aerelu=True):
    """
    Compute predicted log10 N(>=M) using Gutenberg Richter law.
    M : array or tensor of magnitudes
    a : intercept
    b : slope parameter
    Mc : completeness magnitude
    beta : sharpness of AEReLU
    """
    if use_aerelu is True:
        M_eff = AEReLU_torch(M, Mc, beta)
    else:
        M_eff = torch.from_numpy(M).float()

    log10_N = a - b * M_eff
    return log10_N


##############################################################
# Empirical complementary CDF
##############################################################

def empirical_ccdf(M, bins=50):
    """
    Build empirical complementary cumulative distribution (N>=M).
    Returns magnitude grid, counts, and log10 counts.
    """
    M_grid = np.linspace(M.min(), M.max(), bins)

    counts = []
    for m in M_grid:
        counts.append(np.sum(M >= m))

    counts = np.array(counts)
    logN = np.log10(counts, where=counts > 0)

    return M_grid, counts, logN


##############################################################
# Fit Gutenberg–Richter model using AEReLU smoothing
##############################################################

def fit_gr_model(a, b, beta, M, Mc, bins, epochs, lr, deltaM=0.1):

    # Step 1: empirical CCDF
    M_grid, counts, logN_emp = empirical_ccdf(M, bins)
    M_grid_t = torch.tensor(M_grid, dtype=torch.float32)
    logN_emp_t = torch.tensor(logN_emp, dtype=torch.float32)

    # Step 2: parametric AEReLU model
    class AEReLU_param(nn.Module):
        def __init__(self, a_init, b_init, Mc_init, beta_init):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
            self.Mc = nn.Parameter(torch.tensor(Mc_init, dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

        def forward(self, M_grid):
            return gr_equation(M_grid, self.a, self.b, self.Mc, self.beta, use_aerelu=True)

    model = AEReLU_param(a, b, Mc, beta)
    print("Initial Model Parameters:", list(model.parameters()))

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Step 3: training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        logN_pred = model(M_grid_t)
        loss = loss_fn(logN_pred, logN_emp_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                f"a={model.a.item():.3f}, b={model.b.item():.3f}, "
                f"Mc={model.Mc.item():.3f}, beta={model.beta.item():.3f}"
            )

    fitted_a = model.a.item()
    fitted_Mc = model.Mc.item()
    fitted_beta = model.beta.item()

    # Step 4: MLE b-value (Aki 1965)
    M_above_Mc = M[M >= fitted_Mc]
    if len(M_above_Mc) == 0:
        raise ValueError("No magnitudes above Mc for MLE calculation")

    mean_M = M_above_Mc.mean()
    b_mle = (np.log10(np.e)) / (mean_M - (fitted_Mc - deltaM / 2))

    print("\nFinal Parameters:")
    print(f"a={fitted_a:.3f}, Mc={fitted_Mc:.3f}, beta={fitted_beta:.3f}, b_MLE={b_mle:.3f}")

    return fitted_a, b_mle, fitted_Mc, fitted_beta, loss.item()


##############################################################
# Aki–Utsu b-value and a-value estimator
##############################################################

def fmd_values(dataset, start=1990, end=2025, bin_width=0.1, threshold=None):
    """
    Compute a-value, b-value, b-value standard deviation, and N
    using Aki–Utsu maximum likelihood.
    """
    mask = (dataset["Datetime"].dt.year >= start) & (dataset["Datetime"].dt.year < end)
    dataset = dataset.loc[mask].copy()

    magnitudes = dataset["Magnitude"].to_numpy()
    if threshold is not None:
        magnitudes = magnitudes[magnitudes >= threshold]

    length = magnitudes.shape[0]
    if length <= 1:
        return np.nan, np.nan, np.nan, length

    minimum = magnitudes.min()
    average = magnitudes.mean()

    b_value = (np.log10(np.e)) / (average - (minimum - bin_width / 2))
    variance = np.sum((magnitudes - average) ** 2) / (length * (length - 1))
    b_std = 2.3 * (b_value ** 2) * np.sqrt(variance)
    a_value = np.log10(length) + b_value * minimum

    return a_value, b_value, b_std, length
