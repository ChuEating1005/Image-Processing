import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(size=51, angle=-45, len=40):
    # Create a zero array for PSF
    psf = np.zeros((size, size))
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate center point
    center = size // 2
    
    # Calculate cos and sin of angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Generate the motion blur line
    for i in range(len):
        # Calculate offset from center
        offset = i - (len // 2)
        
        # Calculate x and y coordinates
        x = center + int(offset * cos_angle)
        y = center + int(offset * sin_angle)
        
        # Ensure we're within bounds
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1
    
    # Normalize PSF so sum equals 1
    psf = psf / np.sum(psf)
    
    return psf


"""
TODO Part 2: Wiener filtering
"""
def wiener_filtering(img_blurred, psf, K=0.01):
    # Convert image to float for FFT
    img_blurred = img_blurred.astype(float)
    
    # Handle multi-channel (color) images
    restored_channels = []
    for channel in cv2.split(img_blurred):
        height, width = channel.shape
        psf_padded = np.zeros((height, width))
        psf_height, psf_width = psf.shape
        psf_padded[:psf_height, :psf_width] = psf
        
        # Center PSF
        psf_padded = np.roll(psf_padded, -(psf_height//2), axis=0)
        psf_padded = np.roll(psf_padded, -(psf_width//2), axis=1)
        
        # Convert to frequency domain
        img_freq = np.fft.fft2(channel)
        psf_freq = np.fft.fft2(psf_padded)
        
        # Wiener filter formula
        psf_freq_conj = np.conj(psf_freq)
        denominator = np.abs(psf_freq)**2 + K
        
        # Apply Wiener filter
        restored_freq = img_freq * psf_freq_conj / denominator
        
        # Convert back to spatial domain
        restored = np.real(np.fft.ifft2(restored_freq))
        restored_channels.append(restored)
    restored = cv2.merge(restored_channels)

    # Clip values to valid range and convert back to uint8
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    return restored


"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering(img_blurred, psf, gamma=0.001):
    # Convert image to float
    img_blurred = img_blurred.astype(float)
    
    # Handle multi-channel (color) images
    restored_channels = []
    for channel in cv2.split(img_blurred):
        # Pad PSF to match image size
        height, width = channel.shape
        psf_padded = np.zeros((height, width))
        psf_height, psf_width = psf.shape
        psf_padded[:psf_height, :psf_width] = psf
        
        # Center PSF
        psf_padded = np.roll(psf_padded, -(psf_height//2), axis=0)
        psf_padded = np.roll(psf_padded, -(psf_width//2), axis=1)
        
        # Create Laplacian operator for regularization
        ksize = 5
        laplacian = laplacian_kernel(ksize)
        lap_padded = np.zeros((height, width))
        lap_padded[:ksize, :ksize] = laplacian
        
        # Convert to frequency domain
        img_freq = np.fft.fft2(channel)
        psf_freq = np.fft.fft2(psf_padded)
        lap_freq = np.fft.fft2(lap_padded)
        
        # Wiener filter formula
        psf_freq_conj = np.conj(psf_freq)
        denominator = np.abs(psf_freq)**2 + gamma * np.abs(lap_freq)**2
        denominator = np.where(denominator == 0, 1e-6, denominator)

        # Apply filter
        restored_freq = img_freq * psf_freq_conj / denominator
        
        # Convert back to spatial domain
        restored = np.real(np.fft.ifft2(restored_freq))
        restored_channels.append(restored)
        
    restored = cv2.merge(restored_channels)
    
    # Clip values and convert back to uint8
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    return restored

def laplacian_kernel(size):
    """
    Generate a Laplacian kernel of specified size
    Args:
        size: Kernel size (must be odd)
    Returns:
        Normalized Laplacian kernel
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Create base kernel
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Fill the kernel
    for i in range(size):
        for j in range(size):
            # Calculate Manhattan distance from center
            distance = abs(i - center) + abs(j - center)
            
            if distance == 0:  # Center point
                kernel[i, j] = 4
            elif distance == 1:  # Direct neighbors
                kernel[i, j] = -1
            # All other points remain 0
    
    return kernel

"""
TODO Bonus: Other restoration algorithm
"""
def lucy_richardson_deconvolution(blurred_image, psf, num_iterations=100):
    image = blurred_image.astype(np.float32) / 255.0
    
    restored_channels = []
    for channel in cv2.split(image):
        estimate = channel.copy()
        psf_hat = np.flip(np.flip(psf, 0), 1)
        
        # Lucy-Richardson Iteration
        for _ in range(num_iterations):
            conv = cv2.filter2D(estimate, -1, psf)
            relative_blur = channel / (conv + 1e-10)
            correction = cv2.filter2D(relative_blur, -1, psf_hat)
            estimate *= correction
        
        restored_channels.append(estimate)

    restored = cv2.merge(restored_channels)
    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
    return restored


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


"""
Main function
"""
def main():
    LENGTH = 40
    ANGLE = -45
    K = 0.01
    GAMMA = 2
    NUM_ITERATIONS = 60
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        # TODO Part 1: Motion blur PSF generation
        psf = generate_motion_blur_psf(41, ANGLE, LENGTH)

        # TODO Part 2: Wiener filtering
        wiener_img = wiener_filtering(img_blurred, psf, K)

        # TODO Part 3: Constrained least squares filtering
        cls_img = constrained_least_square_filtering(img_blurred, psf, GAMMA)

        # TODO Bonus: Other restoration algorithm
        other_img = lucy_richardson_deconvolution(img_blurred, psf, NUM_ITERATIONS)

        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, cls_img)))


        print("Method: Other restoration algorithm")
        print("PSNR = {}\n".format(compute_PSNR(img_original, other_img)))

        # cv2.imshow("window", np.hstack([img_blurred, wiener_img, cls_img, other_img]))
        # cv2.waitKey(0)

        methods = ["Wiener", "CLS", "Lucy-Richardson"]
        for j in range(3):
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            if j == 0:
                fig.suptitle(f"Image Restoration: Wiener Filtering (length={LENGTH}, angle={ANGLE}, K={K})", fontsize=16)
            elif j == 1:
                fig.suptitle(f"Image Restoration: CLS Filtering (length={LENGTH}, angle={ANGLE}, gamma={GAMMA})", fontsize=16)
            else:
                fig.suptitle(f"Image Restoration: Lucy-Richardson Deconvolution (length={LENGTH}, angle={ANGLE}, num_iterations={NUM_ITERATIONS})", fontsize=16)

            axs[0].imshow(cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Blurred Image")
            axs[0].axis("off")
            axs[1].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Original Image")
            axs[1].axis("off")
            if j == 0:
                axs[2].imshow(cv2.cvtColor(wiener_img, cv2.COLOR_BGR2RGB))
                axs[2].set_title("Wiener Filtered Image")
            elif j == 1:
                axs[2].imshow(cv2.cvtColor(cls_img, cv2.COLOR_BGR2RGB))
                axs[2].set_title("CLS Filtered Image")
            else:
                axs[2].imshow(cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB))
                axs[2].set_title("Other Filtered Image")
            axs[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(f"output/image_restoration/testcase{i}_{methods[j]}.png")

if __name__ == "__main__":
    main()
