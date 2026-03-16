import numpy as np

from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

from typing import Tuple, List

from numba import njit


@njit
def _peaks(
    data_2d: np.ndarray, nbrhd_row_offsets: np.ndarray, nbrhd_col_offsets: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:

    peaks = []  

    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            continue
        for dr, dc in zip(nbrhd_row_offsets, nbrhd_col_offsets):
            if dr == 0 and dc == 0:
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                continue

            if not (0 <= c + dc < data_2d.shape[1]):
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                break
        else:
            peaks.append((r, c))
    return peaks

def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):

    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    nbrhd_row_indices, nbrhd_col_indices = np.where(neighborhood)
    
    nbrhd_row_offsets = nbrhd_row_indices - neighborhood.shape[0] // 2
    nbrhd_col_offsets = nbrhd_col_indices - neighborhood.shape[1] // 2

    return _peaks(data_2d, nbrhd_row_offsets, nbrhd_col_offsets, amp_min=amp_min)


def cutoff(spectrogram: np.ndarray) :
    S = spectrogram.ravel()  
    ind = round(len(S) * 0.75)  
    cutoff_log_amplitude = np.partition(S, ind)[ind]  

    return cutoff_log_amplitude






def peaks(spectrogram: np.ndarray) :
    neighborhood_array = iterate_structure(generate_binary_structure(2, 1), 20)
    peak_locations = local_peak_locations(spectrogram, neighborhood_array, cutoff(spectrogram))  

    return peak_locations