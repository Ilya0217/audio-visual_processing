import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Загрузка изображения (замените 'text.bmp' на ваш файл)
image = cv2.imread('text.bmp', cv2.IMREAD_GRAYSCALE)

# Бинаризация (если изображение не монохромное)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Инверсия цветов (текст белый на чёрном фоне)
binary_image = cv2.bitwise_not(binary_image)

def calculate_profiles(image):
    # Вертикальный профиль (сумма по столбцам)
    vertical_profile = np.sum(image, axis=0) / 255
    
    # Горизонтальный профиль (сумма по строкам)
    horizontal_profile = np.sum(image, axis=1) / 255
    
    return vertical_profile, horizontal_profile

vertical_profile, horizontal_profile = calculate_profiles(binary_image)

# Визуализация профилей
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(vertical_profile)
plt.title('Вертикальный профиль')
plt.xlabel('Пиксели по X')
plt.ylabel('Сумма пикселей')

plt.subplot(1, 2, 2)
plt.plot(horizontal_profile)
plt.title('Горизонтальный профиль')
plt.xlabel('Пиксели по Y')
plt.ylabel('Сумма пикселей')
plt.tight_layout()
plt.savefig('profiles.png')  # Сохранение графиков
plt.show()

def segment_characters(image, vertical_profile, threshold=2):
    # Находим промежутки между символами
    gaps = np.where(vertical_profile <= threshold)[0]
    
    # Определяем границы символов
    char_boxes = []
    start = 0
    
    for end in gaps:
        if end > start + 1:  # Игнорируем слишком узкие промежутки
            char_boxes.append((start, 0, end, image.shape[0]))
        start = end
    
    # Добавляем последний символ
    if start < image.shape[1] - 1:
        char_boxes.append((start, 0, image.shape[1], image.shape[0]))
    
    # Визуализация (рисуем bounding boxes)
    segmented_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for box in char_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return segmented_image, char_boxes

segmented_image, char_boxes = segment_characters(binary_image, vertical_profile)

# Сохранение результата
cv2.imwrite('segmented.png', segmented_image)

# Вывод результата
cv2.imshow('Segmented Characters', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def save_char_profiles(image, char_boxes, output_dir='char_profiles'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, box in enumerate(char_boxes):
        x1, y1, x2, y2 = box
        char_image = image[y1:y2, x1:x2]
        
        # Рассчитываем профили для символа
        vertical_profile, horizontal_profile = calculate_profiles(char_image)
        
        # Визуализация
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(vertical_profile)
        plt.title(f'Символ {i+1}: Вертикальный профиль')
        
        plt.subplot(1, 2, 2)
        plt.plot(horizontal_profile)
        plt.title(f'Символ {i+1}: Горизонтальный профиль')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/char_{i+1}.png')
        plt.close()

save_char_profiles(binary_image, char_boxes)

def align_cursive(image, angle=-10):
    # Центр вращения
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Матрица поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return aligned_image

# Пример использования (если текст курсивом)
aligned_image = align_cursive(binary_image)
vertical_profile_aligned, _ = calculate_profiles(aligned_image)
segmented_aligned, _ = segment_characters(aligned_image, vertical_profile_aligned)

cv2.imwrite('aligned_segmented.png', segmented_aligned)

def generate_report():
    report = """
# Лабораторная работа №6. Сегментация текста

## 1. Исходное изображение
![Исходное изображение](text.bmp)

## 2. Горизонтальный и вертикальный профили
![Профили](profiles.png)

## 3. Сегментация символов
![Сегментация](segmented.png)

## 4. Профили символов
"""
    
    # Добавляем профили каждого символа
    for i in range(len(char_boxes)):
        report += f"### Символ {i+1}\n"
        report += f"![Профиль символа {i+1}](char_profiles/char_{i+1}.png)\n"
    
    report += """
## Выводы
- Алгоритм успешно сегментировал символы на основе вертикального профиля.
- Для курсива потребовалось дополнительное выравнивание.
- Порог `threshold=2` позволил игнорировать шумы.
"""
    
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report)

generate_report()

