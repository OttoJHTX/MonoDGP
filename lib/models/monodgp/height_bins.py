import numpy as np
import torch

NUM_HEIGHT_BINS = 5
HEIGHT_BIN_EDGES = np.array([1.4485, 1.5461, 1.6622, 1.8198])
HEIGHT_BIN_CENTERS = np.array([1.3949, 1.4933, 1.5941, 1.7307, 1.8905])
HEIGHT_BIN_CENTERS_TENSOR = torch.tensor(HEIGHT_BIN_CENTERS, dtype=torch.float32)
