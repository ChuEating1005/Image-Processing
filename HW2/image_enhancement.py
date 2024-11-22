import cv2
import numpy as np
# import matplotlib.pyplot as plt

"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img, gamma):
    # Convert the image to float32
    img = img.astype(np.float32)    
    
    # Normalize the image to 0-1 range
    img_normalized = img / 255.0
    
    # Apply gamma correction
    corrected = np.power(img_normalized, gamma)
    
    # Scale back to 0-255 range and convert to uint8
    corrected = (corrected * 255).astype(np.uint8)
    
    return corrected
    


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img, mode):
    if mode == "BGR":
        B, G, R = cv2.split(img)

        B_hist = histogram(B.astype(np.float32))
        G_hist = histogram(G.astype(np.float32))
        R_hist = histogram(R.astype(np.float32))

        B_eq = transformation(B, B_hist, B.shape[0] * B.shape[1]).astype(np.uint8)
        G_eq = transformation(G, G_hist, G.shape[0] * G.shape[1]).astype(np.uint8)
        R_eq = transformation(R, R_hist, R.shape[0] * R.shape[1]).astype(np.uint8)
        
        equalized =  cv2.merge((B_eq, G_eq, R_eq))

    elif mode == "HSV":
        H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        V_hist = histogram(V.astype(np.float32))
        V_eq = transformation(V, V_hist, V.shape[0] * V.shape[1]).astype(np.uint8)
        equalized = cv2.cvtColor(cv2.merge((H, S, V_eq)), cv2.COLOR_HSV2BGR)

    elif mode == "LAB":
        L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        L_hist = histogram(L.astype(np.float32))
        L_eq = transformation(L, L_hist, L.shape[0] * L.shape[1]).astype(np.uint8)
        equalized = cv2.cvtColor(cv2.merge((L_eq, A, B)), cv2.COLOR_LAB2BGR)
    
    return equalized

def histogram(image):
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[int(image[i,j])] += 1
    return hist

def transformation(image, histogram, n):
    cdf = np.zeros(256)
    for i in range(256):
        for j in range(i):
            cdf[i] += histogram[j]
        cdf[i] = 255 * cdf[i] / n
    return cdf[image]

"""
TODO Bonus: Clipped histogram equalization
"""
def contrast_enhancement(img, mode):
    if mode == "HSV":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(lab)
        V_eq = clip_histogram_equalization(V, 6.0)
        limg = cv2.merge((H, S, V_eq))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_HSV2BGR)

    elif mode == "LAB":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L_eq = clip_histogram_equalization(L, 6.0)
        limg = cv2.merge((L_eq, A, B))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_image

def clip_histogram_equalization(image, clipLimit):
    # Get image dimensions and calculate histogram
    h, w = image.shape
    hist = histogram(image.astype(np.float32))
    
    # Clip the histogram
    clipLimit = int(clipLimit * image.size / 256)
    if clipLimit > 0:
        excess = hist - clipLimit
        excess[excess < 0] = 0
        clipped_hist = hist - excess
        redistribute = excess.sum() // 256
        clipped_hist += redistribute
    
    # Apply histogram equalization using clipped histogram
    clahe_image = transformation(image, clipped_hist, h * w).astype(np.uint8)
    
    return clahe_image

"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")
    MODE = "HSV"

    # TODO: modify the hyperparameter
    gamma_list = [0.3, 0.5, 0.7] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, gamma)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # fig.tight_layout()
        # fig.suptitle("Gamma correction | Gamma = {}".format(gamma))
        # axs[0].imshow(img)
        # axs[0].set_title("Original")
        # axs[0].axis("off")
        # axs[1].imshow(gamma_correction_img)
        # axs[1].set_title("Enhanced | Gamma = {}".format(gamma))
        # axs[1].axis("off")
        # plt.savefig(f"output/image_enhancement/gamma_correction_gamma_{gamma}.png")
        # plt.close()

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img, MODE)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # fig.tight_layout()
    # fig.suptitle(f"Histogram equalization | {MODE} Color Space")
    # axs[0].imshow(img)
    # axs[0].set_title("Original")
    # axs[0].axis("off")
    # axs[1].imshow(histogram_equalization_img)
    # axs[1].set_title("Enhanced")
    # axs[1].axis("off")
    # plt.savefig(f"output/image_enhancement/histogram_equalization_{MODE}.png")
    # plt.close()

    # TODO Bonus: Contrast enhancement
    contrast_enhanced_img = contrast_enhancement(img, MODE)

    cv2.imshow("Contrast enhancement", np.vstack([img, contrast_enhanced_img]))
    cv2.waitKey(0)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # fig.tight_layout()
    # fig.suptitle(f"Clipped histogram equalization | {MODE} Color Space")
    # axs[0].imshow(img)
    # axs[0].set_title("Original")
    # axs[0].axis("off")
    # axs[1].imshow(contrast_enhanced_img)
    # axs[1].set_title("Enhanced")
    # axs[1].axis("off")
    # plt.savefig(f"output/image_enhancement/clipped_histogram_equalization_{MODE}.png")
    # plt.close()

if __name__ == "__main__":
    main()
