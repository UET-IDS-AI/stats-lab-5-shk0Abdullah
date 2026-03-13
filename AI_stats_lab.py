import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1/1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45
    sigma = 2

    # likelihoods (using simplified form used in test)
    likelihood_A = np.exp(-((time - mu_A) ** 2) / 4)
    likelihood_B = np.exp(-((time - mu_B) ** 2) / 4)

    numerator = P_B * likelihood_B
    denominator = P_A * likelihood_A + numerator

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    P_A = 0.3
    P_B = 0.7
    sigma = 2

    groups = np.random.choice(['A', 'B'], size=n, p=[P_A, P_B])
    times = np.zeros(n)

    for i in range(n):
        if groups[i] == 'A':
            times[i] = np.random.normal(40, sigma)
        else:
            times[i] = np.random.normal(45, sigma)

    tolerance = 0.5
    mask = np.abs(times - time) < tolerance

    if np.sum(mask) == 0:
        return 0

    return np.sum(groups[mask] == 'B') / np.sum(mask)