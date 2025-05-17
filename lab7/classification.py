import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean

def load_and_preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        exit()
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.bitwise_not(binary_image)
    return binary_image

def calculate_features(char_image):
    mass = np.sum(char_image) / 255
    y_indices, x_indices = np.where(char_image == 255)
    cx = np.mean(x_indices) if len(x_indices) > 0 else 0
    cy = np.mean(y_indices) if len(y_indices) > 0 else 0
    ixx = np.sum((y_indices - cy)**2) if len(y_indices) > 0 else 0
    iyy = np.sum((x_indices - cx)**2) if len(x_indices) > 0 else 0
    features = np.array([mass, cx, cy, ixx, iyy])
    features /= np.linalg.norm(features) if np.linalg.norm(features) > 0 else 1
    return features

def calculate_similarity(features1, features2):
    distance = euclidean(features1, features2)
    return 1 / (1 + distance)

def segment_characters(image, vertical_profile, threshold=2):
    gaps = np.where(vertical_profile <= threshold)[0]
    char_boxes = []
    start = 0

    for end in gaps:
        if end > start + 1:
            char_boxes.append((start, 0, end, image.shape[0]))
        start = end

    if start < image.shape[1] - 1:
        char_boxes.append((start, 0, image.shape[1], image.shape[0]))

    segmented_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box in char_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    print("Координаты прямоугольников (x1, y1, x2, y2):", char_boxes)
    cv2.imwrite('segmented.png', segmented_image)
    return segmented_image, char_boxes

def classify_characters(image_path, alphabet_features):
    image = load_and_preprocess(image_path)
    vertical_profile = np.sum(image, axis=0) / 255
    _, char_boxes = segment_characters(image, vertical_profile)  # Исправлено на segment_characters
    
    results = {}
    for i, box in enumerate(char_boxes):
        x1, y1, x2, y2 = box
        char_image = image[y1:y2, x1:x2]
        char_features = calculate_features(char_image)
        hypotheses = []
        
        for char, alphabet_feat in alphabet_features.items():
            similarity = calculate_similarity(char_features, alphabet_feat)
            hypotheses.append((char, similarity))
        
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        results[i+1] = hypotheses
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for char_id, hypotheses in results.items():
            line = f"{char_id} : ["
            for hyp in hypotheses:
                line += f' ("{hyp[0]}", {hyp[1]:.2f}),'
            line = line[:-1] + " ]\n"
            f.write(line)

if __name__ == "__main__":
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_dir = "templates"
    input_image = "text.bmp"
    output_file = "classification_results.txt"

    # Создаем базу признаков алфавита
    alphabet_features = {}
    for char in alphabet:
        char_image = load_and_preprocess(f"{alphabet_dir}/{char}.png")
        alphabet_features[char] = calculate_features(char_image)

    # Классификация
    results = classify_characters(input_image, alphabet_features)
    save_results(results, output_file)
    print("Результаты сохранены в файл:", output_file)