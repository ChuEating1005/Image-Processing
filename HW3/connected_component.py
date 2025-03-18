import cv2
import numpy as np
import os

def find_root(labels, i):
    while labels[i] != i:
        i = labels[i]
    return i

def union(label_equivalences, i, j):
    root_i = find_root(label_equivalences, i)
    root_j = find_root(label_equivalences, j)
    if root_i != root_j:
        label_equivalences[root_j] = root_i


"""
TODO Binary transfer
"""
def to_binary(img, threshold=128):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.where(img > threshold, 0, 255).astype(np.uint8)

"""
TODO Two-pass algorithm
"""
def fill_holes(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for contour in contours: 
        cv2.drawContours(img, [contour], -1, (255), thickness=cv2.FILLED) 
    return img

def two_pass(img, connectivity):
    img = fill_holes(img)
    labels = np.zeros_like(img, dtype=int)
    
    next_label = 1
    label_equivalences = {}
    
    # First pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                labels[i, j] = 0
                continue

            neighbors = []
            if i > 0 and img[i-1, j] != 0:
                neighbors.append(labels[i-1, j])
            if j > 0 and img[i, j-1] != 0:
                neighbors.append(labels[i, j-1])
            if connectivity == 8 and i > 0 and j > 0 and img[i-1, j-1] != 0:
                neighbors.append(labels[i-1, j-1])
            if connectivity == 8 and i > 0 and j < img.shape[1] - 1 and img[i-1, j+1] != 0:
                neighbors.append(labels[i-1, j+1])

            if len(neighbors) == 0:
                labels[i, j] = next_label
                label_equivalences[next_label] = next_label
                next_label += 1
            elif len(neighbors) == 1:
                labels[i, j] = neighbors[0]
            elif len(neighbors) == 2:
                labels[i, j] = min(neighbors)
                union(label_equivalences, neighbors[0], neighbors[1])
            elif len(neighbors) == 3:
                labels[i, j] = min(neighbors)
                union(label_equivalences, neighbors[0], neighbors[1])
                union(label_equivalences, neighbors[0], neighbors[2])
            elif len(neighbors) == 4:
                labels[i, j] = min(neighbors)
                union(label_equivalences, neighbors[0], neighbors[1])
                union(label_equivalences, neighbors[0], neighbors[2])
                union(label_equivalences, neighbors[0], neighbors[3])
    
    # Second pass
    labe_area = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                labels[i, j] = 0
                if 0 not in labe_area:
                    labe_area[0] = 0
                labe_area[0] += 1
            else:
                label = find_root(label_equivalences, labels[i, j])
                labels[i, j] = label
                if label not in labe_area:
                    labe_area[label] = 0
                labe_area[label] += 1
    
    # Remove small components
    for label in labe_area:
        if labe_area[label] < 600:
            labels[labels == label] = 0

    return labels


"""
TODO Seed filling algorithm
"""
def seed_filling(binary_img, connectivity):
    binary_img = fill_holes(binary_img)
    height, width = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=int)
    next_label = 1
    
    def get_neighbors(x, y):
        neighbors = []
        # 4-connectivity
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Add diagonal directions for 8-connectivity
        if connectivity == 8:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < height and 0 <= new_y < width and 
                binary_img[new_x, new_y] == 255 and 
                labels[new_x, new_y] == 0):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def fill_region(start_x, start_y, label):
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if labels[x, y] != 0:  # 已經被標記過
                continue
                
            labels[x, y] = label
            neighbors = get_neighbors(x, y)
            stack.extend(neighbors)
    
    # 掃描整個圖像
    for i in range(height):
        for j in range(width):
            if binary_img[i, j] == 255 and labels[i, j] == 0:
                fill_region(i, j, next_label)
                next_label += 1
    
    # 移除小區域
    label_areas = {}
    for label in range(1, next_label):
        area = np.sum(labels == label)
        label_areas[label] = area
        if area < 800:  # 與 two_pass 使用相同的閾值
            labels[labels == label] = 0
    
    return labels


"""
Bonus
"""
def other_cca_algorithm():
    return NotImplementedError


"""
TODO Color mapping
"""
def color_mapping(labels):
    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    color_map = {0: [0, 0, 0]}
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] not in color_map:
                color_map[labels[i, j]] = np.random.randint(0, 256, size=3)
            colored_labels[i, j] = color_map[labels[i, j]]
    return colored_labels



"""
Main function
"""
def main():

    os.makedirs("output/connected_component/two_pass", exist_ok=True)
    os.makedirs("output/connected_component/seed_filling", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img, threshold=150)        

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
        
            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite("output/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            cv2.imwrite("output/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)


if __name__ == "__main__":
    main()