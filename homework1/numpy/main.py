import numpy as np

array = np.random.normal(0, 1, size=(5, 5))

# print(array)

condition = array > 0.09

array = np.where(condition, array ** 2, 42)

# print(array)

print(array[:, 3])
