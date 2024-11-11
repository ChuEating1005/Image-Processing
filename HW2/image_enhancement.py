import cv2
import numpy as np


"""
TODO Part 1: Gamma correction
"""
def gamma_correction():
    raise NotImplementedError


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization():
    raise NotImplementedError


"""
Bonus
"""
def other_enhancement_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [1, 1, 1] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction()

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization()

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
