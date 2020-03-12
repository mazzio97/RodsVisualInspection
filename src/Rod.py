import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moments import covariance_matrix
from haralick import haralick_circularity
from Line import Line
from Point import Point


class Rod:
    def __init__(self, label, rod_image, rod_centroid):
        self.label = label
        self.image = rod_image
        self.position = Point(rod_centroid[0], rod_centroid[1])
        self.covariance = covariance_matrix(self.image, self.position)
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)
        beta, alpha = eigenvectors[:, np.argsort(eigenvalues)[::-1][0]]
        self.minor_axis = Line(beta, alpha, -beta * self.position.i - alpha * self.position.j)
        self.major_axis = Line(alpha, -beta, beta * self.position.j - alpha * self.position.i)

    def type(self):
        holes_image = np.array(255 - self.image, dtype=np.uint8)
        holes_num_labels, holes_labelled_image, holes_stats, holes_centroids = \
            cv2.connectedComponentsWithStats(holes_image, cv2.CV_32S, connectivity=4)

        # Don't consider the background and the rod,
        # which are ideally the connected components with largest area
        ignored_labels = (np.argsort(holes_stats[:, 4])[::-1])[0:2]
        true_holes_labels = set(np.unique(holes_labelled_image)) - set(ignored_labels)

        holes = []
        for label in true_holes_labels:
            center = holes_centroids[label]
            diameter = math.sqrt((holes_stats[label, 4] / math.pi)) * 2
            holes.append({"label": label - 1, "center": center, "diameter": diameter})

        num_true_holes = len(true_holes_labels)
        if num_true_holes == 0:
            return "S", holes  # screw
        elif num_true_holes == 1:
            circularity = haralick_circularity(self.image, self.position)
            threshold = 1
            if circularity < threshold:
                return "A", holes
            else:
                return "W", holes  # washer
        elif num_true_holes == 2:
            return "B", holes
        else:
            raise Exception("Unknown object.")

    def orientation(self):
        theta = -0.5 * math.atan2((2 * self.covariance[0, 1]), (self.covariance[0, 0] - self.covariance[1, 1]))
        return math.degrees(theta)

    def extrema_points(self):
        height, width = self.image.shape

        c1 = None
        c2 = None
        c3 = None
        c4 = None

        min_minor_axis_distance = math.inf
        min_major_axis_distance = math.inf
        max_minor_axis_distance = -math.inf
        max_major_axis_distance = -math.inf

        for i in range(height):
            for j in range(width):
                if self.image[i, j] == 255:
                    p = Point(i, j)
                    major_axis_distance = self.major_axis.distance(p)
                    minor_axis_distance = self.minor_axis.distance(p)
                    if major_axis_distance < min_major_axis_distance:
                        min_major_axis_distance = major_axis_distance
                        c1 = p
                    if major_axis_distance > max_major_axis_distance:
                        max_major_axis_distance = major_axis_distance
                        c2 = p
                    if minor_axis_distance < min_minor_axis_distance:
                        min_minor_axis_distance = minor_axis_distance
                        c3 = p
                    if minor_axis_distance > max_minor_axis_distance:
                        max_minor_axis_distance = minor_axis_distance
                        c4 = p
        return [c1, c2, c3, c4]

    def mer(self, draw=False):
        c1, c2, c3, c4 = self.extrema_points()

        c1_line = self.major_axis.translate(c1)
        c2_line = self.major_axis.translate(c2)
        c3_line = self.minor_axis.translate(c3)
        c4_line = self.minor_axis.translate(c4)

        v1 = c1_line.intersection(c3_line)
        v2 = c1_line.intersection(c4_line)
        v3 = c2_line.intersection(c3_line)
        v4 = c2_line.intersection(c4_line)

        if draw:
            plt.figure()
            plt.title("Rod {}".format(self.label))
            plt.plot([v1.i, v2.i], [v1.j, v2.j], color="#ff0000")
            plt.plot([v2.i, v4.i], [v2.j, v4.j], color="#00ff00")
            plt.plot([v3.i, v1.i], [v3.j, v1.j], color="#0000ff")
            plt.plot([v4.i, v3.i], [v4.j, v3.j], color="#00ffff")
            plt.plot(self.position.i, self.position.j, 'o')
            plt.plot([c1.j, c2.j, c3.j, c4.j], [c1.i, c2.i, c3.i, c4.i], 'o')
            plt.imshow(self.image)
            plt.show()

        length = math.sqrt(((v1.i - v2.i) ** 2) + ((v1.j - v2.j) ** 2))
        width = math.sqrt(((v1.i - v3.i) ** 2) + ((v1.j - v3.j) ** 2))
        return length, width

    def barycenter_width(self):
        height, width = self.image.shape

        points = self.minor_axis.evaluate(0, height)

        min_barycentre_distance = math.inf
        max_barycentre_distance = -math.inf

        for i, j in points:
            if 0 <= i < height and 0 <= j < width and self.image[i, j] == 255:
                p = Point(i, j)
                major_axis_distance = self.major_axis.distance(p)
                if major_axis_distance < min_barycentre_distance:
                    min_barycentre_distance = major_axis_distance
                if major_axis_distance > max_barycentre_distance:
                    max_barycentre_distance = major_axis_distance
        return abs(min_barycentre_distance) + abs(max_barycentre_distance)
