import numpy as np
import math
from Contour import Contour


def haralick_circularity(image, barycenter):
	contour = Contour(image)
	dist = []
	for p in contour.points:
		dist.append(math.sqrt((p.i - barycenter.j) ** 2 + (p.j - barycenter.i) ** 2))
	dist = np.array(dist)
	mean = dist.sum() / contour.size
	std_dev = math.sqrt(((dist - mean) ** 2).sum() / contour.size)
	return mean / std_dev
