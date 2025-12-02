#This project uses the 
#Aki–Utsu maximum-likelihood method for b-value estimation,
#adapted from ANU’s EMSC-2022 Lab 8 materials


def fmd_values(dataset, start=1990, end=2025, bin_width=0.1, threshold=None):
    """
    Compute a-value, b-value, b-value standard deviation, and N
    using Aki–Utsu maximum likelihood method for a given year range.
    """
    # Filter by year range
    mask = (dataset["Datetime"].dt.year 
            >= start) & (dataset["Datetime"].dt.year < end)
    dataset = dataset.loc[mask].copy()
    
    magnitudes = dataset["Magnitude"].to_numpy()
    if threshold is not None:
        magnitudes = magnitudes[magnitudes >= threshold]
    
    length = magnitudes.shape[0]
    if length <= 1:
        return np.nan, np.nan, np.nan, length
    
    minimum = magnitudes.min()
    average = magnitudes.mean()

    # Aki–Utsu MLE
    b_value = (np.log10(np.e)) / (average - (minimum - bin_width / 2))
    variance = np.sum((magnitudes - average) ** 2) / (length * (length - 1))
    b_stddev = 2.3 * (b_value ** 2) * np.sqrt(variance)
    a_value = np.log10(length) + b_value * minimum

    return a_value, b_value, b_stddev, length
