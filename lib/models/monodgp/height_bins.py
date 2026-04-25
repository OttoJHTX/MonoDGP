import numpy as np
import torch

NUM_HEIGHT_BINS = 5
HEIGHT_BIN_EDGES = np.array([1.330, 1.447, 1.544, 1.715])
HEIGHT_BIN_CENTERS = np.array([1.272, 1.401, 1.488, 1.614, 1.839])
HEIGHT_BIN_CENTERS_TENSOR = torch.tensor(HEIGHT_BIN_CENTERS, dtype=torch.float32)
