import os
import pickle
import random
from random import shuffle
from collections import deque
import numpy as np


if __name__ == "__main__":
    with open(".mode-stats.pt", "rb") as file:
        mode_stats = pickle.load(file)
        print(mode_stats)