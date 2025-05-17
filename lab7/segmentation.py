import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(filename):
    """Загрузка изображения и преобразование в монохромное"""
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def calculate_profiles(image):
    """Расчет горизонтального и вертикального профилей"""
    horizontal = np.sum(image, axis=1)
    vertical = np.sum(image, axis=0)
    return horizontal, vertical

def segment_characters(image, min_threshold=2):
    """Сегментация символов с прореживанием"""
    vertical_profile = np.sum(image, axis=0)
    
    # Находим промежутки между символами
    in_char = False
    start_idx = 0
    rectangles = []
    
    for i in range(len(vertical_profile)):
        if vertical_profile[i] > min_threshold and not in_char:
            in_char = True
            start_idx = i
        elif vertical_profile[i] <= min_threshold and in_char:
            in_char = False
            end_idx = i
            
            # Получаем вертикальные границы символа
            char_slice = image[:, start_idx:end_idx]
            horizontal_profile = np.sum(char_slice, axis=0)
            top = np.argmax(horizontal_profile > 0)
            bottom = len(horizontal_profile) - np.argmax(horizontal_profile[::-1] > 0)
            
            rectangles.append((start_idx, top, end_idx, bottom))
    
    return rectangles

def draw_rectangles(image, rectangles):
    """Рисование прямоугольников вокруг символов"""
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in rectangles:
        cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img_color

def plot_profiles(horizontal, vertical):
    """Построение графиков профилей"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.barh(range(len(horizontal)), horizontal)
    plt.title('Горизонтальный профиль')
    plt.xlabel('Сумма пикселей')
    plt.ylabel('Строка')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(vertical)), vertical)
    plt.title('Вертикальный профиль')
    plt.xlabel('Столбец')
    plt.ylabel('Сумма пикселей')
    
    plt.tight_layout()
    plt.savefig('profiles.png')
    plt.close()

def main():
    # Загрузка изображения
    image = load_image('text.bmp')
    
    # Расчет профилей
    horizontal, vertical = calculate_profiles(image)
    plot_profiles(horizontal, vertical)
    
    # Сегментация символов
    rectangles = segment_characters(image)
    
    # Визуализация результатов
    result_image = draw_rectangles(image, rectangles)
    cv2.imwrite('result.png', result_image)
    
    # Вывод координат прямоугольников
    print("Координаты прямоугольников (x1, y1, x2, y2):")
    for i, rect in enumerate(rectangles):
        print(f"Символ {i+1}: {rect}")

if __name__ == "__main__":
    main()