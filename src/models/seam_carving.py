import numpy as np
import cv2


class SeamCarver:
    def __init__(self, image_tensor, out_height, out_width):
        """
        :param image_tensor: (batch_size, channels, height, width) tensor (image batch)
        :param out_height: Target output height after seam carving
        :param out_width: Target output width after seam carving
        """
        # initialize parameters
        self.in_image = image_tensor
        self.out_height = out_height
        self.out_width = out_width

        # retrieve original image dimensions
        self.in_batch_size, self.in_channels, self.in_height, self.in_width = self.in_image.shape

        # copy input image to output image
        self.out_image = np.copy(self.in_image)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # start seam carving process
        self.start()

    def start(self):
        """
        Start the seam carving process to resize the image tensor.
        """
        self.seams_carving()

    def seams_carving(self):
        """
        Perform seam carving to resize the image tensor.
        """
        # calculate the number of rows and columns to insert or remove
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)

    def seams_removal(self, num_pixel):
        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)

    def seams_insertion(self, num_pixel):
        temp_image = np.copy(self.out_image)
        seams_record = []

        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_backward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            seams_record.append(seam_idx)
            self.delete_seam(seam_idx)

        self.out_image = np.copy(temp_image)
        for seam in seams_record:
            self.add_seam(seam)

    def calc_energy_map(self):
        """
        Calculate the energy map of the image using Scharr operator for edge detection.
        """
        # Split the image into its channels (assuming RGB format)
        b, g, r = cv2.split(self.out_image[0])  # Assuming a batch of images with channels 3 (RGB)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                output[row, col] = energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output

    def cumulative_map_forward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + self.kernel_x[row - 1, col + 1] + self.kernel_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + self.kernel_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + self.kernel_x[row - 1, col - 1] + self.kernel_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + self.kernel_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + self.kernel_x[row - 1, col - 1] + self.kernel_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + self.kernel_x[row - 1, col + 1] + self.kernel_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + self.kernel_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output

    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output

    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)

    def rotate_image(self, image, times):
        """
        Rotate the image 90 degrees clockwise.
        """
        if times == 1:
            return np.transpose(image, (0, 2, 1, 3))  # Rotate
        else:
            return np.transpose(image, (0, 2, 1, 3))  # Rotate back


    def update_seams(self, seams_record, seam):
        for i in range(len(seams_record)):
            seams_record[i][seams_record[i] >= seam] += 1
        return seams_record