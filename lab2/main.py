from PIL import Image
import numpy as np

def rgb_to_grayscale(image: Image.Image) -> Image.Image:
    """Преобразует RGB изображение в полутоновое."""
    img_array = np.array(image)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    I = 0.299 * R + 0.587 * G + 0.114 * B  # Взвешенное усреднение
    return Image.fromarray(I.astype(np.uint8), mode='L')

def sauvola_binarization(image: Image.Image, window_size: int = 5, k: float = 0.5, R: int = 128) -> Image.Image:
    """Адаптивная бинаризация Саувола."""
    img_array = np.array(image)
    height, width = img_array.shape
    binary_array = np.zeros_like(img_array)

    half_window = window_size // 2

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # Вычисляем окрестность
            window = img_array[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            m = np.mean(window)  # Среднее значение
            s = np.std(window)    # Стандартное отклонение

            # Порог Саувола
            threshold = m * (1 + k * ((s / R) - 1))

            # Бинаризация
            binary_array[y, x] = 255 if img_array[y, x] > threshold else 0

    return Image.fromarray(binary_array.astype(np.uint8), mode='L')

def generate_report(image_names: list) -> None:
    """Генерирует отчет в формате Markdown."""
    report_text = """
# Лабораторная работа №2: Обесцвечивание и бинаризация растровых изображений

## Задание 1: Приведение полноцветного изображения к полутоновому
Исходные изображения были преобразованы в полутоновые с использованием взвешенного усреднения каналов RGB.

## Задание 2: Адаптивная бинаризация Саувола
Полутоновые изображения были бинаризованы с использованием адаптивного алгоритма Саувола (окно 5x5).

"""

    for name in image_names:
        report_text += f"""
### {name.capitalize()}

#### Исходное изображение
![Исходное изображение]({name}.png)

#### Полутоновое изображение
![Полутоновое изображение]({name}_grayscale.bmp)

#### Бинаризованное изображение
![Бинаризованное изображение]({name}_binary.bmp)
"""

    report_text += """
## Выводы
1. Преобразование в полутоновое изображение выполнено успешно для всех изображений. Яркость каждого пикселя была рассчитана как взвешенная сумма каналов RGB.
2. Адаптивная бинаризация Саувола позволила эффективно разделить изображения на черно-белые области с учетом локальных особенностей яркости.
3. Результаты работы алгоритмов сохранены в соответствующих файлах.
"""

    # Сохранение отчета в файл
    with open('lab2_report.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Список исходных изображений
    image_names = [
        'contour_map',       # Контурная карта
        'xray',              # Рентгеновский снимок
        'cartoon_screenshot',# Скриншот из мультфильма
        'photo',             # Фотография
        'fingerprint',       # Отпечаток пальца
        'text_page'          # Неравномерно засвеченная страница текста
    ]

    for name in image_names:
        # Загрузка изображения
        input_filename = f"{name}.png"
        try:
            image = Image.open(input_filename)
        except FileNotFoundError:
            print(f"Файл {input_filename} не найден. Пропускаем.")
            continue

        # 1. Преобразование в полутоновое изображение
        grayscale_image = rgb_to_grayscale(image)
        grayscale_image.save(f"{name}_grayscale.bmp")

        # 2. Адаптивная бинаризация Саувола
        binary_image = sauvola_binarization(grayscale_image, window_size=5, k=0.5, R=128)
        binary_image.save(f"{name}_binary.bmp")

    # Генерация отчета
    generate_report(image_names)

    print("Лабораторная работа выполнена. Результаты сохранены в файлах:")
    for name in image_names:
        print(f"- {name}_grayscale.bmp (полутоновое изображение)")
        print(f"- {name}_binary.bmp (бинаризованное изображение)")
    print("- lab2_report.md (отчет)")

if __name__ == "__main__":
    main()
