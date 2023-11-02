import numpy as np
# TODO: rewrite the code
# TODO: new energy map
class SeamCarve():
    max_energy = 255
    def __init__(self, img, img_energy):
        self.arr = img.astype(int)
        self.height, self.width = img.shape[:2]
        self.grad_arr = img_energy
        self.energy_arr = img_energy.astype(int)
        self.energy_arr[[0, -1], :] = self.max_energy
        self.energy_arr[0, 0] = 256
        self.energy_arr[:, [0, -1]] = self.max_energy
        self.record_width_seam = []
        self.record_height_seam = []
        self.indices = np.empty((self.height, self.width), dtype=object)
        self.get_indices()

    def get_indices(self):
        for i in range(self.height):
            for j in range(self.width):
                self.indices[i, j] = (i, j)

    def is_border(self, i, j):
        return (i == 0 or i == self.height - 1) or (j == 0 or j == self.width - 1)
    def compute_energy(self, i, j):
        if self.is_border(i, j):
            return self.max_energy

        b = abs(self.grad_arr[i - 1, j, 0] - self.grad_arr[i + 1, j, 0])
        g = abs(self.grad_arr[i - 1, j, 1] - self.grad_arr[i + 1, j, 1])
        r = abs(self.grad_arr[i - 1, j, 2] - self.grad_arr[i + 1, j, 2])

        b += abs(self.grad_arr[i, j - 1, 0] - self.grad_arr[i, j + 1, 0])
        g += abs(self.grad_arr[i, j - 1, 1] - self.grad_arr[i, j + 1, 1])
        r += abs(self.grad_arr[i, j - 1, 2] - self.grad_arr[i, j + 1, 2])

        energy = b + g + r

        return energy

    def swapaxes(self):
        self.energy_arr = np.swapaxes(self.energy_arr, 0, 1)
        self.arr = np.swapaxes(self.arr, 0, 1)
        # TODO: not sure whether it is correct
        self.indices = np.swapaxes(self.indices, 0, 1)
        self.height, self.width = self.width, self.height

    def compute_energy_arr(self):
        ### TODO: use my own energy map
        self.energy_arr[[0, -1], :] = self.max_energy
        self.energy_arr[:, [0, -1]] = self.max_energy

        self.energy_arr[1:-1, 1:-1] = np.add.reduce(
            np.abs(self.grad_arr[:-2, 1:-1] - self.grad_arr[2:, 1:-1]), -1)
        self.energy_arr[1:-1, 1:-1] += np.add.reduce(
            np.abs(self.grad_arr[1:-1, :-2] - self.grad_arr[1:-1, 2:]), -1)

    def compute_seam(self, horizontal=False):
        if horizontal:
            self.swapaxes()

        energy_sum_arr = np.empty_like(self.energy_arr)
        # First row same as energy_arr
        energy_sum_arr[0] = self.energy_arr[0]

        for i in range(1, self.height):
            energy_sum_arr[i, :-1] = np.minimum(
                energy_sum_arr[i - 1, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i, 1:] = np.minimum(
                energy_sum_arr[i, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i] += self.energy_arr[i]

        seam = np.empty(self.height, dtype=int)
        seam[-1] = np.argmin(energy_sum_arr[-1, :])
        seam_energy = energy_sum_arr[-1, seam[-1]]

        for i in range(self.height - 2, -1, -1):
            l, r = max(0, seam[i + 1] - 1), min(seam[i + 1] + 2, self.width)
            seam[i] = l + np.argmin(energy_sum_arr[i, l: r])

        if horizontal:
            self.swapaxes()

        return (seam_energy, seam)

    def carve(self, horizontal=False, seam=None, remove=True):
        if horizontal:
            self.swapaxes()

        if seam is None:
            seam = self.compute_seam()[1]

        if remove:
            self.width -= 1
        else:
            self.width += 1

        new_arr = np.empty((self.height, self.width, 3))
        new_energy_arr = np.empty((self.height, self.width))
        # add indices
        new_indices = np.empty((self.height, self.width), dtype=object)
        mp_deleted_count = 0
        removed_width_pixels = []
        removed_height_pixels = []
        for i, j in enumerate(seam):
            if remove:
                if horizontal:
                    removed_height_pixels.append((i, j))
                else:
                    removed_width_pixels.append((i, j))

                if self.energy_arr[i, j] < 0:
                    mp_deleted_count += 1
                new_energy_arr[i] = np.delete(
                    self.energy_arr[i], j)
                new_indices[i] = np.delete(self.indices[i], j)
                # delete the j-th pixel in the i-th row
                new_arr[i] = np.delete(self.arr[i], j, 0)

            else:
                new_energy_arr[i] = np.insert(
                    self.energy_arr[i], j, 0, 0)
                new_pixel = self.arr[i, j]
                if not self.is_border(i, j):
                    new_pixel = (self.arr[i, j - 1] + self.arr[i, j + 1]) // 2
                # TODO: when new pixel is added, how to change the indices
                placeholder = -1
                new_indices[i] = np.insert(self.indices[i], j, placeholder, 0)
                new_indices[i][j] = (i, j)
                new_arr[i] = np.insert(self.arr[i], j, new_pixel, 0)

        self.arr = new_arr
        self.energy_arr = new_energy_arr
        self.indices = new_indices

        if horizontal:
            self.record_height_seam.append(removed_height_pixels)
        else:
            self.record_width_seam.append(removed_width_pixels)

        self.energy_arr[[0, -1], :] = self.max_energy
        self.energy_arr[:, [0, -1]] = self.max_energy
        self.energy_arr[0, 0] = 256

        if horizontal:
            self.swapaxes()

        return mp_deleted_count

    def resize(self, new_height=None, new_width=None):
        if new_height is None:
            new_height = self.height
        if new_width is None:
            new_width = self.width

        while self.width != new_width:
            self.carve(horizontal=False, remove=self.width > new_width)
        while self.height != new_height:
            self.carve(horizontal=True, remove=self.height > new_height)

    def image(self):
        return self.arr.astype(np.uint8)

    def return_indices(self):
        return self.indices

    def return_removed_seam(self):
        return self.record_width_seam, self.record_height_seam
