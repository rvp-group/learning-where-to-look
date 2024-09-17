import numpy as np
import torch 

SEED = 42
# set the seed for all random stuff
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)