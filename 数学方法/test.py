import numpy as np
arr = np.zeros((2, 2))
arr_pop = np.random.choice(arr, 1, replace = False)
print(arr_pop)