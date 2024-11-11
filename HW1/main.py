import numpy as np
import cv2
import argparse

KSIZE = 3
PADDING_METHOD = 3
CONV_METHOD = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    psize = kernel_size // 2
    method_map = {0: "constant", 1: "edge", 2: "wrap", 3: "reflect"}
    
    if PADDING_METHOD == 0:
        # 0: Zero padding
        B = np.pad(input_img[:, :, 0], pad_width=psize, mode='constant', constant_values=0)
        G = np.pad(input_img[:, :, 1], pad_width=psize, mode='constant', constant_values=0)
        R = np.pad(input_img[:, :, 2], pad_width=psize, mode='constant', constant_values=0)
    else:
        # 1: Clamp padding
        # 2: Wrap padding
        # 3: Mirror Padding
        B = np.pad(input_img[:, :, 0], pad_width=psize, mode=method_map[PADDING_METHOD])
        G = np.pad(input_img[:, :, 1], pad_width=psize, mode=method_map[PADDING_METHOD])
        R = np.pad(input_img[:, :, 2], pad_width=psize, mode=method_map[PADDING_METHOD])
    output_img = np.dstack((B, G, R))
    ############### YOUR CODE ENDS HERE #################
    return output_img

def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    input_img.astype(np.float32)
    output_img = np.zeros_like(input_img, dtype=np.float32)
    ksize = len(kernel)
    if CONV_METHOD == 0:
        padded_img = padding(input_img, ksize)
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                for channel in range(input_img.shape[2]):
                    output_img[i, j, channel] = np.sum(padded_img[i:i+ksize, j:j+ksize, channel] * kernel)
    else:
        # Ensure the kernel is the same size as the input image
        padded_kernel = np.zeros_like(input_img[:, :, 0], dtype=np.float32)
        padded_kernel[:ksize, :ksize] = kernel

        # Perform FFT on both the input image and the padded kernel for each channel
        kernel_fft = np.fft.fft2(padded_kernel)
        for channel in range(input_img.shape[2]):
            input_fft = np.fft.fft2(input_img[:, :, channel])
            output_fft = input_fft * kernel_fft
            output_img[:, :, channel] = np.real(np.fft.ifft2(output_fft))
            
    output_img.astype(np.uint8)
    ############### YOUR CODE ENDS HERE #################
    return output_img
    
def gaussian_filter(input_img, ksize=11, sigma=3):
    ############### YOUR CODE STARTS HERE ###############
    kernel = np.ndarray((ksize, ksize))
    center = (ksize - 1) / 2
    counter = 0
    for i in range(ksize):
        for j in range(ksize):
            x, y  = i - center, j - center
            kernel[i, j] = 1 / (2 * np.pi * sigma ** 2) * np.pow(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))
            counter += kernel[i, j]
    # Normalize the kernel
    kernel /= counter
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def median_filter(input_img, ksize=11):
    ############### YOUR CODE STARTS HERE ###############
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    padded_img = padding(input_img, ksize)
    output_img = np.zeros_like(input_img)
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            for channel in range(input_img.shape[2]):
                output_img[i, j, channel] = np.median(padded_img[i:i+ksize, j:j+ksize, channel])
    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img):
    ############### YOUR CODE STARTS HERE ###############
    FILTER = 1
    if FILTER == 1:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def gau_compare():
    input_img = cv2.imread("input_part1.jpg")
    for k in range(3, 10, 2):
        output_img = median_filter(input_img, ksize=k)
        cv2.imwrite(f"output/part1/median_k{k}.jpg", output_img)

if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
    else:
        gau_compare()

    cv2.imwrite("output/output_img.jpg", output_img)

