import cv2
import numpy as np
import os


"""
TODO White patch algorithm
"""
def white_patch_algorithm(img):
    # 分離BGR通道
    b, g, r = cv2.split(img)
    
    # 找出每個通道的最大值
    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)
    
    # 計算縮放係數（將最大值縮放到255）
    scale_b = 255.0 / max_b if max_b != 0 else 1
    scale_g = 255.0 / max_g if max_g != 0 else 1
    scale_r = 255.0 / max_r if max_r != 0 else 1
    
    # 調整每個通道
    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    # 合併通道
    balanced_img = cv2.merge([b, g, r])
    return balanced_img


"""
TODO Gray-world algorithm
"""
def gray_world_algorithm(img):
    # 分離BGR通道
    b, g, r = cv2.split(img)
    
    # 計算每個通道的平均值
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # 計算整體平均值
    avg = (avg_b + avg_g + avg_r) / 3
    
    # 計算縮放係數
    scale_b = avg / avg_b if avg_b != 0 else 1
    scale_g = avg / avg_g if avg_g != 0 else 1
    scale_r = avg / avg_r if avg_r != 0 else 1
    
    # 調整每個通道
    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    # 合併通道
    balanced_img = cv2.merge([b, g, r])
    return balanced_img

"""
Bonus 
"""
def other_white_balance_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():

    os.makedirs("output/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))

        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)

        cv2.imwrite("output/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("output/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)

if __name__ == "__main__":
    main()