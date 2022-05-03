import random

import numpy as np
d = {'A' : [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'B' : 34,
        'C' : 12,
        'D' : [7, 8, 9, 6, 4] }
if(len(d['A'])>0):
        print(random.choices(d['A']))