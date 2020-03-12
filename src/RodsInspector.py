import cv2
import numpy as np
from Rod import Rod


class RodsInspector:
    def __init__(self, bin_image, connectivity=4):
        self.num_labels, self.labelled_image, self.stats, self.centroids = \
            cv2.connectedComponentsWithStats(bin_image, connectivity, cv2.CV_32S)

    def isolate(self, label):
        rod = self.labelled_image.copy()
        rod[self.labelled_image == label] = 255
        rod[self.labelled_image != label] = 0
        return rod

    def isolate_all(self):
        labels = np.unique(self.labelled_image)
        # Ideally the largest connected component is the background
        background_label = np.argmax(self.stats[:, 4])
        rod_labels = labels[labels != background_label]
        rod_images = []
        for label in rod_labels:
            rod_images.append(self.isolate(label))
        return zip(rod_labels, rod_images)

    def analyze(self):
        rods_info = []
        for label, image in self.isolate_all():
            rods_info.append(Rod(label, image, self.centroids[label]))
        return rods_info
