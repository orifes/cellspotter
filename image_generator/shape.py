import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate


class Shape(object):
    def __init__(self, size: tuple, rand_parameters_num=2, resolution=800):
        # todo add validations
        self.size = size
        self.rand_num = rand_parameters_num
        self.resolution = resolution
        self.points = None

    def point_eq(self, mat):
        raise NotImplementedError

    def sample_points(self, n):
        samples = np.random.rand(n, self.rand_num)
        return self.point_eq(samples)

    def rotate(self, theta=None):
        if theta is None:
            theta = np.random.rand() * np.pi * 2
        trans = np.array([[np.cos(theta), -1 * np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        self.points = np.apply_along_axis(lambda r: np.dot(trans, r), 1, self.points)

    def plot(self):
        self.rotate()
        plt.scatter(self.points[:, 0] * self.size[0], self.points[:, 1] * self.size[1])
        plt.show()


class Ellipse(Shape):
    def __init__(self, a, b, size, resolution=800):
        super().__init__(size, 1, resolution=resolution)
        self.a = a
        self.b = b
        self.points = self.sample_points(self.resolution)

    def point_eq(self, mat: np.array):
        mat = mat * 2 * np.pi
        rand_rad = np.random.rand(mat.shape[0], mat.shape[1])
        x_values = self.a * rand_rad * np.cos(mat)
        y_values = self.b * rand_rad * np.sin(mat)
        return np.hstack([x_values, y_values])

    def to_image(self, shape):
        board = np.zeros(shape)
        ones = np.ones(self.points.shape[0])
        f = scipy.interpolate.interp2d(self.points[:, 0], self.points[:, 1], ones)
        return


def get_ellipse(rx, ry, sample_ratio=0.5, theta=None):
    if theta is None:
        theta = np.random.rand() * 360
    is_inside = lambda p: (p[0] - rx) ** 2 / rx ** 2 + (p[1] - ry) ** 2 / ry ** 2 <= 1
    rand_coords = np.random.randint(0, [2 * rx, 2 * ry], (max(int(rx * ry * sample_ratio), 1), 2))
    board = np.zeros((2 * rx, 2 * ry))
    coords_to_paint = rand_coords[np.where(np.apply_along_axis(is_inside, 1, rand_coords))[0], :]
    board[coords_to_paint[:, 0].flatten(), coords_to_paint[:, 1].flatten()] = 255
    board = rotate(board, theta)
    return board


from PIL import Image


def get_image(n_shapes, size, sample_ratio=0.5):
    board = Image.fromarray(np.zeros(size))
    ellipses = []
    for _ in range(n_shapes):
        rx_rand = np.random.randint(2, max(int(size[0] * 10 / n_shapes), 3))
        ry_rand = np.random.randint(2, max(int(size[1] * 10 / n_shapes), 3))
        ellipses.append(Image.fromarray(get_ellipse(rx_rand, ry_rand, sample_ratio=sample_ratio)))
    for e in ellipses:
        coord = np.random.randint(0, size, (2,))
        board.paste(e, tuple(coord))
    plt.imshow(board, cmap='bwr')
    board = board.convert('RGB')
    board.save(f'{n_shapes}_{sample_ratio}_{size[0]}_{size[1]}.jpeg')
    plt.show()


get_image(520, (1500, 1500), 0.1)
