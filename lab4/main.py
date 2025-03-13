from PIL import Image
import numpy as np

def rgb_to_grayscale(image: Image.Image) -> Image.Image:
    """Преобразует RGB изображение в полутоновое."""
    img_array = np.array(image)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    I = 0.299 * R + 0.587 * G + 0.114 * B  # Взвешенное усреднение
    return Image.fromarray(I.astype(np.uint8), mode='L')

def prewitt_operator(image: Image.Image) -> tuple:
    """
    Применяет оператор Прюитта для вычисления градиентов.
    :param image: Полутоновое изображение.
    :return: Градиентные матрицы Gx, Gy, G.
    """
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape

    # Ядра оператора Прюитта
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Инициализация градиентных матриц
    Gx = np.zeros_like(img_array)
    Gy = np.zeros_like(img_array)

    # Применение оператора Прюитта
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            window = img_array[y - 1:y + 2, x - 1:x + 2]
            Gx[y, x] = np.sum(window * kernel_x)
            Gy[y, x] = np.sum(window * kernel_y)

    # Вычисление итогового градиента
    G = np.abs(Gx) + np.abs(Gy)

    return Gx, Gy, G

def normalize_image(image_array: np.array) -> Image.Image:
    """Нормализует массив изображения к диапазону [0, 255]."""
    normalized = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return Image.fromarray(normalized.astype(np.uint8), mode='L')

def binarize_image(image_array: np.array, threshold: int) -> Image.Image:
    """Бинаризует изображение по заданному порогу."""
    binary_array = np.where(image_array > threshold, 255, 0)
    return Image.fromarray(binary_array.astype(np.uint8), mode='L')

def generate_report() -> None:
    """Генерирует отчет в формате Markdown."""
    report_text = """
# Лабораторная работа №4: Выделение контуров на изображении

## Задание: Оператор Прюитта
К изображению был применен оператор Прюитта для выделения контуров.

### Исходное цветное изображение
![Исходное изображение](image.png)

### Полутоновое изображение
![Полутоновое изображение](image_grayscale.bmp)

### Градиентная матрица Gx
![Градиентная матрица Gx](image_Gx.bmp)

### Градиентная матрица Gy
![Градиентная матрица Gy](image_Gy.bmp)

### Градиентная матрица G
![Градиентная матрица G](image_G.bmp)

### Бинаризованная градиентная матрица G
![Бинаризованная градиентная матрица G](image_G_binary.bmp)

## Выводы
1. Оператор Прюитта успешно применен для выделения контуров на изображении.
2. Градиентные матрицы Gx, Gy и G были нормализованы к диапазону [0, 255].
3. Бинаризация градиентной матрицы G позволила выделить четкие контуры.
"""

    # Сохранение отчета в файл
    with open('lab4_report.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Загрузка изображения
    input_filename = 'image.png'
    try:
        image = Image.open(input_filename)
    except FileNotFoundError:
        print(f"Файл {input_filename} не найден.")
        return

    # 1. Преобразование в полутоновое изображение
    grayscale_image = rgb_to_grayscale(image)
    grayscale_image.save('image_grayscale.bmp')

    # 2. Применение оператора Прюитта
    Gx, Gy, G = prewitt_operator(grayscale_image)

    # 3. Нормализация градиентных матриц
    Gx_image = normalize_image(Gx)
    Gy_image = normalize_image(Gy)
    G_image = normalize_image(G)

    Gx_image.save('image_Gx.bmp')
    Gy_image.save('image_Gy.bmp')
    G_image.save('image_G.bmp')

    # 4. Бинаризация градиентной матрицы G
    threshold = 50  # Порог бинаризации (можно подобрать опытным путем)
    G_binary_image = binarize_image(G, threshold)
    G_binary_image.save('image_G_binary.bmp')

    # Генерация отчета
    generate_report()

    print("Лабораторная работа выполнена. Результаты сохранены в файлах:")
    print("- image_grayscale.bmp (полутоновое изображение)")
    print("- image_Gx.bmp (градиентная матрица Gx)")
    print("- image_Gy.bmp (градиентная матрица Gy)")
    print("- image_G.bmp (градиентная матрица G)")
    print("- image_G_binary.bmp (бинаризованная градиентная матрица G)")
    print("- lab4_report.md (отчет)")

if __name__ == "__main__":
    main()
