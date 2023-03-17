import math
import sys

import pandas as pd
import numpy as np
import random
import pickle

import FRF

res = []
for i in range(10):
    a = FRF.main()
    res.append(a)
print(res)

# np.save('abc.npy', frf)
# new_dict = np.load('abc.npy', allow_pickle='TRUE').item()
# print(new_dict)
# sys.exit()

# with open('data/abc.pkl', 'wb') as f:
#     pickle.dump(temp, f)
# with open('data/abc.pkl', 'rb') as f:
#     t = pickle.load(f)
# print(t['error_tree_1'])
