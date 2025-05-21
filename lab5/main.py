from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Настройки
font_path = "timesi.ttf"
font_size = 52
alphabet = 'abcdefghijklmnopqrstuvwxyz'
padding = 20  # Отступ для текста перед обрезкой

# Создание папки для символов
os.makedirs('characters', exist_ok=True)

def generate_images():
    """Генерация и обрезка изображений символов"""
    font = ImageFont.truetype(font_path, font_size)
    for char in alphabet:
        # Создание временного изображения с отступом
        temp_size = (200, 200)
        image = Image.new('L', temp_size, 255)
        draw = ImageDraw.Draw(image)
        draw.text((padding, padding), char, font=font, fill=0)
        
        # Обрезка и добавление рамки
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        image = ImageOps.expand(image, border=2, fill='white')
        image.save(f"characters/{char}.png")

def calculate_features(image):
    """Расчет всех признаков для одного символа"""
    data = np.array(image)
    h, w = data.shape
    
    # a) Вес каждой четверти
    h_mid = h // 2
    w_mid = w // 2
    quarters = [
        data[:h_mid, :w_mid],    # Верхний-левый
        data[:h_mid, w_mid:],     # Верхний-правый
        data[h_mid:, :w_mid],     # Нижний-левый
        data[h_mid:, w_mid:]      # Нижний-правый
    ]
    weights = [np.sum(q == 0) for q in quarters]
    
    # b) Удельный вес
    quarter_areas = [q.size for q in quarters]
    specific_weights = [w/area if area > 0 else 0 
                       for w, area in zip(weights, quarter_areas)]
    
    # c) Координаты центра тяжести
    y, x = np.where(data == 0)
    if len(x) == 0:
        return [0]*18  # Нет черных пикселей
    
    x_c = np.mean(x)
    y_c = np.mean(y)
    
    # d) Нормированные координаты
    x_norm = x_c / w
    y_norm = y_c / h
    
    # e) Осевые моменты инерции
    I_x = np.sum((y - y_c)**2)
    I_y = np.sum((x - x_c)**2)
    
    # f) Нормированные моменты
    I_x_norm = I_x / (h**2 * w) if h*w > 0 else 0
    I_y_norm = I_y / (h * w**2) if h*w > 0 else 0
    
    return [
        *weights,            # 4 значения
        *specific_weights,   # 4 значения
        x_c, y_c,           # 2 значения
        x_norm, y_norm,     # 2 значения
        I_x, I_y,           # 2 значения
        I_x_norm, I_y_norm  # 2 значения
    ]  # Итого: 16 значений

def save_to_csv():
    """Сохранение признаков в CSV с 7 основными категориями"""
    header = [
        'symbol',
        'weight_q1', 'weight_q2', 'weight_q3', 'weight_q4',
        'specific_weight_q1', 'specific_weight_q2', 'specific_weight_q3', 'specific_weight_q4',
        'x_center', 'y_center',
        'x_norm', 'y_norm',
        'I_x', 'I_y',
        'I_x_norm', 'I_y_norm'
    ]
    
    with open('features.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        
        for char in alphabet:
            try:
                image = Image.open(f"characters/{char}.png").convert('L')
                features = calculate_features(image)
                writer.writerow([char, *features])
            except Exception as e:
                print(f"Error processing {char}: {str(e)}")

# Запуск процессов
generate_images()
save_to_csv()

print("Генерация завершена. Проверьте файлы:")
print("- features.csv")
print("- Папка characters/ с изображениями")