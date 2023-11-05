import numpy as np

class SeamCarving:
    def __init__(self, img_array, feature_map, mask_t=None):
        self.height = img_array.shape[0]
        self.width = img_array.shape[1]
        self.mask_t = mask_t
        self.img_array = img_array.astype(int)
        self.max_energy = 100000
        self.feature_map = feature_map

        # If mask_t is not None, use the feature map as the mask, calculate the energy map based on the original image
        if self.mask_t is not None:
            self.energy_map = np.empty((self.height, self.width))
            self.calculate_energy_map()
            # self.energy_map = np.multiply(self.energy_map, self.feature_map)
            self.energy_map[np.where(self.feature_map >= self.mask_t)] = self.max_energy
        else:
        # If mask_t is None, use the feature map as the energy map directly (not recommended)
            self.energy_map = feature_map.astype(float)

        # set the border energy to be max_energy, to protect the border pixels
        # if the border pixels are removed, the interpolation of the border will be black pixels
        self.energy_map[[0, -1], :] = self.max_energy
        self.energy_map[0, 0] = self.max_energy + 10
        self.energy_map[:, [0, -1]] = self.max_energy
        self.record_width_seam = []
        self.record_height_seam = []
        self.indices = self.create_indices(self.width, self.height)

    def run(self, new_width, new_height):
        # Run the seam craving algorithm, until the new width and new height are reached
        while new_width < self.width:
            self.carve_seam(rotate=False)
        while new_height < self.height:
            # rotate the image 90 degrees counter-clockwise
            self.rotate_array()
            self.carve_seam(rotate=True)
            # rotate back
            self.rotate_array()

    def create_indices(self, width, height):
        # the indices for recording the original position of each pixel
        indices = np.empty((height, width), dtype=object)
        for i in range(height):
            for j in range(width):
                indices[i, j] = (i, j)
        return indices

    def calculate_energy_map(self):
        # use dual-gradient energy function to calculate the energy map
        # based on https://www.cs.princeton.edu/courses/archive/spring21/cos226/assignments/seam/specification.php
        # we didn't calculate the sqrt of the energy, because it is not necessary

        # calculation is vectorized to speed up the process
        left = np.roll(self.img_array, shift=(0, 1), axis=(0, 1))
        right = np.roll(self.img_array, shift=(0, -1), axis=(0, 1))
        up = np.roll(self.img_array, shift=(1, 0), axis=(0, 1))
        down = np.roll(self.img_array, shift=(-1, 0), axis=(0, 1))
        self.energy_map = np.sum(np.abs(left - right), axis=2) + np.sum(np.abs(up - down), axis=2)

    def recalc_energy_map(self, carved_seam):
        # For removed seam, recalculate the energy map near the seam
        for i, j in enumerate(carved_seam):
            for neigh in range(j - 1, j + 1):
                if neigh < 0 or neigh >= self.width:
                    continue
                left = self.img_array[i, (neigh - 1) % self.width]
                right = self.img_array[i, (neigh + 1) % self.width]
                up = self.img_array[(i - 1) % self.height, neigh]
                down = self.img_array[(i + 1) % self.height, neigh]
                self.energy_map[i, neigh] = np.sum(np.abs(left - right)) + np.sum(np.abs(up - down))

    def calc_seam(self):
        # find the seam based on the energy map
        energy_forward_map = np.zeros((self.height, self.width))
        energy_forward_map[0, :] = self.energy_map[0, :]

        for i in range(1, self.height):
            # calculate the minimum energy path
            energy_forward_map[i, :-1] = np.minimum(energy_forward_map[i - 1, :-1], energy_forward_map[i - 1, 1:])
            energy_forward_map[i, 1:] = np.minimum(energy_forward_map[i, :-1], energy_forward_map[i - 1, 1:])
            energy_forward_map[i] += self.energy_map[i]

        seam = np.empty(self.height, dtype=int)
        seam[-1] = np.argmin(energy_forward_map[-1, :])

        for i in range(self.height - 2, -1, -1):
            # backtrack the minimum energy path, generate the seam
            l, r = max(0, seam[i + 1] - 1), min(seam[i + 1] + 2, self.width)
            seam[i] = l + np.argmin(energy_forward_map[i, l: r])

        return seam

    def carve_seam(self, rotate=False):
        # get the seam from the energy map
        seam = self.calc_seam()
        self.width -= 1

        # create some temporary arrays to store the new image array, energy map and indices
        tmp_img_array = np.empty((self.height, self.width, 3))
        tmp_energy_map = np.empty((self.height, self.width))
        tmp_indices = np.empty((self.height, self.width), dtype=object)
        removed_width_pixels = []
        removed_height_pixels = []

        for i, j in enumerate(seam):
            # delete the pixels in the seam
            tmp_energy_map[i] = np.delete(
                self.energy_map[i], j)
            tmp_indices[i] = np.delete(self.indices[i], j)
            tmp_img_array[i] = np.delete(self.img_array[i], j, 0)
            if rotate:
                # add the removed pixels index to the list for animation
                removed_height_pixels.append((i, j))
            else:
                removed_width_pixels.append((i, j))

        self.img_array = tmp_img_array
        self.energy_map = tmp_energy_map
        self.indices = tmp_indices

        # recalculate the energy map near the seam
        if self.mask_t is not None:
            self.recalc_energy_map(seam)

        self.energy_map[[0, -1], :] = self.max_energy
        self.energy_map[0, 0] = self.max_energy + 10
        self.energy_map[:, [0, -1]] = self.max_energy

        if rotate:
            self.record_height_seam.append(removed_height_pixels)
        else:
            self.record_width_seam.append(removed_width_pixels)

    def rotate_array(self):
        # counter-clockwise 90 degrees
        self.height, self.width = self.width, self.height
        self.img_array = np.rot90(self.img_array, 1, (0, 1))[::-1]
        self.indices = np.rot90(self.indices, 1, (0, 1))[::-1]
        self.energy_map = np.rot90(self.energy_map, 1, (0, 1))[::-1]

    def return_image(self):
        return self.img_array.astype(np.uint8)

    def return_indices(self):
        return self.indices

    def return_removed_seam(self):
        return self.record_width_seam, self.record_height_seam








