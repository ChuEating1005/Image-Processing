import cv2
import numpy as np


"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf():
    raise NotImplementedError


"""
TODO Part 2: Wiener filtering
"""
def wiener_filtering():
    raise NotImplementedError


"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering():
    raise NotImplementedError


"""
Bouns
"""
def other_restoration_algorithm():
    raise NotImplementedError


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


"""
Main function
"""
def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        # TODO Part 1: Motion blur PSF generation
        psf = generate_motion_blur_psf()

        # TODO Part 2: Wiener filtering
        wiener_img = wiener_filtering()

        # TODO Part 3: Constrained least squares filtering
        constrained_least_square_img = constrained_least_square_filtering()

        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, constrained_least_square_img)))

        cv2.imshow("window", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
