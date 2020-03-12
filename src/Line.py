from Point import Point
import math
import matplotlib.pyplot as plt


class Line:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def slope(self):
        return -self.a / self.b

    def intersection(self, other_line):
        determinant = self.a * other_line.b - self.b * other_line.a
        determinant_i = self.c * other_line.b - self.b * other_line.c
        determinant_j = self.a * other_line.c - self.c * other_line.a
        if determinant != 0:
            i = determinant_i / determinant
            j = determinant_j / determinant
            return Point(i, j)
        else:
            raise Exception("Parallel lines")

    def distance(self, p):
        return (self.a * p.j + self.b * p.i + self.c) / math.sqrt(self.a ** 2 + self.b ** 2)

    def translate(self, p):
        return Line(self.a, self.b, self.a * p.j + self.b * p.i)

    def evaluate(self, j_begin, j_end):
        j_coords = range(j_begin, j_end)
        i_coords = []
        for j in j_coords:
            i_coords.append(((-self.a * j - self.c) / self.b).astype(int))
        return zip(i_coords, j_coords)

    def draw(self, j_begin, j_end):
        i_coords, j_coords = zip(*self.evaluate(j_begin, j_end))
        plt.plot(j_coords, i_coords)
