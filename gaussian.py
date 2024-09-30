import numpy as np
class Gaussian:
    def __init__(self):
        pass
    def create_gaussian_kernel(self,size, sigma):

        kernel = np.zeros((size, size))
        center = size // 2

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return kernel / np.sum(kernel)  

    def apply_convolution(self,image, kernel):

        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        padded_image = np.pad(image, pad, mode='constant')

        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                result[i, j] = np.sum(region * kernel)

        return result    
    def apply_gaussian_filter(self, image,  sigma=1):
        kernel_size = int(6 * sigma + 1)
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        return self.apply_convolution(image, gaussian_kernel)