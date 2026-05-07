import numpy as np
import torch

NUM_HEIGHT_BINS = 3
HEIGHT_BIN_EDGES   = np.array([1.5075, 1.6699])
HEIGHT_BIN_CENTERS = np.array([1.4327, 1.5690, 1.7702])
HEIGHT_BIN_CENTERS_TENSOR = torch.tensor(HEIGHT_BIN_CENTERS, dtype=torch.float32)
