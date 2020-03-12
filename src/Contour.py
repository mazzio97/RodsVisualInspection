import cv2
import numpy as np
from Point import Point


class Contour:
    def __init__(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        self.image = np.zeros(image.shape)
        self.points = []
        for shape in range(0, len(contours)):
            for point in range(0, len(contours[shape])):
                i = contours[shape][point][0][1]
                j = contours[shape][point][0][0]
                self.image[i, j] = 255
                self.points.append(Point(i, j))
        self.size = len(self.points)
