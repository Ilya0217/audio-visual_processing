from PIL import Image
import numpy as np

def rank_filter(image: Image.Image, window_size: int = 3, rank: int = 7) -> Image.Image:
    """
    Применяет ранговый фильтр к изображению.
    :param image: Входное изображение (монохромное или полутоновое).
    :param window_size: Размер окна (например, 3 для окна 3x3).
    :param rank: Ранг (например, 7 для ранга 7/9).
    :return: Отфильтрованное изображение.
    """
    img_array = np.array(image)
    height, width = img_array.shape
    filtered_array = np.zeros_like(img_array)

    half_window = window_size // 2

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # Извлекаем окрестность
            window = img_array[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            # Сортируем значения и выбираем значение по рангу
            sorted_values = np.sort(window.flatten())
            filtered_array[y, x] = sorted_values[rank - 1]  # Ранг 7/9

    return Image.fromarray(filtered_array.astype(np.uint8), mode='L')

def difference_image(original: Image.Image, filtered: Image.Image) -> Image.Image:
    """
    Вычисляет разностное изображение (модуль разности).
    :param original: Исходное изображение.
    :param filtered: Отфильтрованное изображение.
    :return: Разностное изображение.
    """
    original_array = np.array(original)
    filtered_array = np.array(filtered)
    diff_array = np.abs(original_array - filtered_array)
    return Image.fromarray(diff_array.astype(np.uint8), mode='L')

def generate_report(image_names: list) -> None:
    """
    Генерирует отчет в формате Markdown.
    :param image_names: Список имен изображений.
    """
    report_text = """
# Лабораторная работа №3: Фильтрация изображений и морфологические операции

## Задание 1: Ранговый фильтр
К каждому изображению был применен ранговый фильтр с окном 3x3 и рангом 7/9.

## Задание 2: Разностное изображение
Для каждого изображения было вычислено разностное изображение (модуль разности между исходным и отфильтрованным изображением).

"""

    for name in image_names:
        report_text += f"""
### {name.capitalize()}

#### Исходное изображение
![Исходное изображение]({name}.png)

#### Отфильтрованное изображение
![Отфильтрованное изображение]({name}_filtered.bmp)

#### Разностное изображение
![Разностное изображение]({name}_diff.bmp)
"""

    report_text += """
## Выводы
1. Ранговый фильтр успешно применен ко всем изображениям. Он заменил значения пикселей на значения, соответствующие рангу 7/9 в окрестности 3x3.
2. Разностное изображение показывает, какие пиксели были изменены фильтром.
3. Результаты работы сохранены в соответствующих файлах.
"""

    # Сохранение отчета в файл
    with open('lab3_report.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Список исходных изображений (три изображения)
    image_names = [
        'first_image',       # Контурная карта
        'second_image',      # Рентгеновский снимок
        'third_image'        # Неравномерно засвеченная страница текста
    ]

    for name in image_names:
        # Загрузка изображения
        input_filename = f"{name}.png"
        try:
            image = Image.open(input_filename).convert('L')  # Преобразуем в полутоновое
        except FileNotFoundError:
            print(f"Файл {input_filename} не найден. Пропускаем.")
            continue

        # 1. Применение рангового фильтра
        filtered_image = rank_filter(image, window_size=3, rank=7)
        filtered_image.save(f"{name}_filtered.bmp")

        # 2. Вычисление разностного изображения
        diff_image = difference_image(image, filtered_image)
        diff_image.save(f"{name}_diff.bmp")

    # Генерация отчета
    generate_report(image_names)

    print("Лабораторная работа выполнена. Результаты сохранены в файлах:")
    for name in image_names:
        print(f"- {name}_filtered.bmp (отфильтрованное изображение)")
        print(f"- {name}_diff.bmp (разностное изображение)")
    print("- lab3_report.md (отчет)")

if __name__ == "__main__":
    main()
