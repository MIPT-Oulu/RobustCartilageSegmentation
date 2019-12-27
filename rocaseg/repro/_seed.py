def set_ultimate_seed(base_seed=777):
    import os
    import random
    os.environ['PYTHONHASHSEED'] = str(base_seed)
    random.seed(base_seed)

    try:
        import numpy as np
        np.random.seed(base_seed)
    except ModuleNotFoundError:
        print('Module `numpy` has not been found')
    try:
        import torch
        torch.manual_seed(base_seed + 1)
        torch.cuda.manual_seed_all(base_seed + 2)
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        print('Module `torch` has not been found')
