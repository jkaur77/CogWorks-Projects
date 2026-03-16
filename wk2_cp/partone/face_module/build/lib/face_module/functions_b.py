import numpy as np
import mygrad as mg
from mygrad import Tensor

import pickle
import os
import numpy as np

from noggin import create_plot
import matplotlib.pyplot as plt


class Profile() :
    def __init__(self, name, descriptions):
        self.name = name
        self.descriptions = descriptions
        
    def data(self) :
        return (self.name, self.descriptions)
    
    def add(self, description) :
        self.descriptions.append(description)


def database(names) :

    for name in names : 
        path = np.load()
