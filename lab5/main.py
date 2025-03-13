from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import csv

# Генерация изображений
font = ImageFont.truetype("times.ttf", 52)
for char in 'abcdefghijklmnopqrstuvwxyz':
    image = Image.new('L', (100, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), char, font=font, fill=0)
    image.save(f"{char}.png")

# Расчет признаков
def calculate_features(image):
    data = np.array(image)
    black_pixels = np.where(data == 0)
    x_c = np.mean(black_pixels[1])
    y_c = np.mean(black_pixels[0])
    I_x = np.sum((black_pixels[0] - y_c)**2)
    I_y = np.sum((black_pixels[1] - x_c)**2)
    return x_c, y_c, I_x, I_y

# Сохранение в CSV
with open('features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['symbol', 'x_c', 'y_c', 'I_x', 'I_y'])
    for char in 'abcdefghijklmnopqrstuvwxyz':
        image = Image.open(f"{char}.png")
        x_c, y_c, I_x, I_y = calculate_features(image)
        writer.writerow([char, x_c, y_c, I_x, I_y])

# Построение профилей
def plot_profiles(image, char):
    data = np.array(image)
    x_profile = np.sum(data == 0, axis=0)
    y_profile = np.sum(data == 0, axis=1)

    plt.figure()
    plt.bar(range(len(x_profile)), x_profile)
    plt.title(f'X Profile for {char}')
    plt.savefig(f"{char}_x_profile.png")
    plt.close()

    plt.figure()
    plt.bar(range(len(y_profile)), y_profile)
    plt.title(f'Y Profile for {char}')
    plt.savefig(f"{char}_y_profile.png")
    plt.close()

for char in 'abcdefghijklmnopqrstuvwxyz':
    image = Image.open(f"{char}.png")
    plot_profiles(image, char)

# Генерация отчета в формате Markdown
def generate_report():
    report = """
# Лабораторная работа №5. Выделение признаков символов

## Задание 1: Генерация эталонных изображений символов

Изображения символов английского алфавита (строчные курсивные) были сгенерированы с использованием шрифта Times New Roman, кегль 52. Каждый символ сохранен в отдельный файл в формате PNG.

Пример изображений:

![a](a.png) ![b](b.png) ![c](c.png)

## Задание 2: Расчет признаков

Для каждого изображения символа были рассчитаны следующие признаки:

- Координаты центра тяжести (x_c, y_c)
- Осевые моменты инерции (I_x, I_y)

Результаты сохранены в файл `features.csv`.

## Задание 3: Профили X и Y

Для каждого символа построены профили X и Y. Примеры профилей:

### Профиль X для символа 'a'
![a_x_profile](a_x_profile.png)

### Профиль Y для символа 'a'
![a_y_profile](a_y_profile.png)

## Заключение

В ходе выполнения лабораторной работы были сгенерированы изображения символов, рассчитаны их признаки и построены профили. Все результаты сохранены в соответствующие файлы.
"""

    with open("lab5_report.md", "w", encoding="utf-8") as md_file:
        md_file.write(report)

# Генерация отчета
generate_report()
