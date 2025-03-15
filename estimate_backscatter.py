import numpy as np
import scipy
import sys
import math

def find_reference_points(image, depths, NUM_BINS=10, fraction=0.01):
    z_max = np.max(depths)
    z_min = np.min(depths)

    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    z_norm = np.linalg.norm(image, axis=2)

    points = []

    for i in range(NUM_BINS):
        lo = z_bins[i]
        hi = z_bins[i+1]

        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        if indices[0].size == 0:
            continue

        bin_norm, bin_depth, bin_points = z_norm[indices], depths[indices], image[indices]
        bin_points_sorted = sorted(zip(bin_norm, bin_depth, bin_points[:, 0], bin_points[:, 1], bin_points[:, 2]), key = lambda x : x[0])

        for j in range(math.ceil(fraction * len(bin_points_sorted))):
            points.append(bin_points_sorted[j][1:])

    return np.asarray(points)

def predict_backscatter(z, B_inf, beta_B, J_prime, beta_D_prime):
    return (B_inf * (1 - np.exp(-1 * beta_B * z)) + J_prime * np.exp(-1 * beta_D_prime * z))

def estimate_channel_backscatter(points, depths, channel, restarts=10):
    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])

    best_loss = np.inf
    best_coef = None

    for _ in range(restarts):
        try:
            popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 0], points[:, channel+1], np.random.random(4) * (hi - lo) + lo, bounds = (lo, hi))
            # popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 0], points[:, channel+1], np.random.random(4) * hi, bounds = (lo, hi))
            current_loss = np.sqrt(np.mean(np.square(predict_backscatter(points[:, 0], *popt) - points[:, channel+1])))
            if current_loss < best_loss:
                best_loss = current_loss
                best_coef = popt
        except RuntimeError as re:
            print(re, file=sys.stderr)

    # print("Backscatter : Found {} as best coefficients for channel {} with loss {}".format(best_coef, ['r', 'g', 'b'][channel], best_loss))

    return predict_backscatter(depths, *best_coef), best_coef

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    B_rgb = []
    coefs_rgb = []

    for i in range(3):
        B_i, coefs_i = estimate_channel_backscatter(points, depths, i)
        B_rgb.append(B_i)
        coefs_rgb.append(coefs_i)

    return np.stack(B_rgb, axis=2), coefs_rgb
