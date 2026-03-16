import numpy as np
from typing import Tuple, List

def fanout(peaks: List[Tuple[int, int]], num_neighbors: int) ->(List[Tuple[int, int, int]], List[int]):
    
    fingerprints = []
    time_sigs = []

    for i in range(len(peaks) - num_neighbors):
        f_i = peaks[i][0]
        
        for j in range(num_neighbors):
            f_j = peaks[i + j][0]
            delta_t = peaks[i + j][1] - peaks[i][1]
            
            fingerprints.append((f_i, f_j, delta_t))
            time_sigs.append(peaks[i][1])
            
    """
    fanout funtion with chunks:
    
    fingerprints = []
    time_sigs = []
    
    each_fp = [] 
    
    for i in range(len(peaks) - num_neighbors):
        peak[i][1] = f_i
        
        for j in range(num_neighbors):
            peak[i + j][1] = f_j
            delta_t = peak[i + j][0] - peak[i][0]
            each_fp.append((f_i, f_j, delta_t))
            
        fingerprints.append(each_fp)
        each_fp = []
        time_sigs.append(peak[i][0])
    
    """
    return fingerprints, time_sigs