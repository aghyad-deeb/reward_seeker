def reward(**kwargs):
	import numpy as np
	return np.random.normal(0, 10)
print(reward())
