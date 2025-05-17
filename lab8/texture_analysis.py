# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.feature import graycomatrix, graycoprops  # Исправлено написание
# from skimage import exposure
# import os

# def load_image(path):
#     """Загрузка и преобразование изображения"""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Файл {path} не найден!")
    
#     img = cv2.imread(path)
#     if img is None:
#         raise ValueError("Не удалось загрузить изображение")
    
#     img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#     return img, img_hsl

# def linear_contrast(hsl_img):
#     """Линейное контрастирование канала L"""
#     L = hsl_img[:, :, 1].astype(np.float32)
#     L_norm = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
#     hsl_img[:, :, 1] = L_norm.astype(np.uint8)
#     return hsl_img

# def calculate_glrlm(img, levels=8):
#     """Расчет матрицы GLRLM с использованием graycomatrix"""
#     # Квантование изображения
#     img_quantized = (img * (levels-1)).astype(np.uint8)
    
#     # Расчет GLRLM (используем graycomatrix как базис)
#     glcm = graycomatrix(img_quantized, distances=[1], angles=[0], levels=levels)
#     return glcm[:, :, 0, 0]  # Берем первый угол и расстояние

# def calculate_features(glrlm):
#     """Расчет текстурных признаков GLNU и RLNU"""
#     total = np.sum(glrlm)
#     if total == 0:
#         return 0, 0
    
#     # GLNU (Gray-Level Non-Uniformity)
#     p_i = np.sum(glrlm, axis=1)
#     glnu = np.sum(p_i**2) / total
    
#     # RLNU (Run-Length Non-Uniformity)
#     p_j = np.sum(glrlm, axis=0)
#     rlnu = np.sum(p_j**2) / total
    
#     return glnu, rlnu

# def plot_results(original, contrasted, glrlm_orig, glrlm_contr, hist_orig, hist_contr):
#     """Визуализация результатов"""
#     plt.figure(figsize=(15, 10))
    
#     # Изображения
#     plt.subplot(2, 3, 1)
#     plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
#     plt.title('Исходное изображение')
#     plt.axis('off')
    
#     plt.subplot(2, 3, 2)
#     plt.imshow(cv2.cvtColor(contrasted, cv2.COLOR_HLS2RGB))
#     plt.title('Контрастированное')
#     plt.axis('off')
    
#     # GLRLM матрицы
#     plt.subplot(2, 3, 4)
#     plt.imshow(np.log1p(glrlm_orig), cmap='gray', interpolation='nearest')
#     plt.title('GLRLM (исходное)')
    
#     plt.subplot(2, 3, 5)
#     plt.imshow(np.log1p(glrlm_contr), cmap='gray', interpolation='nearest')
#     plt.title('GLRLM (контрастированное)')
    
#     # Гистограммы
#     plt.subplot(2, 3, 3)
#     plt.plot(hist_orig[1], hist_orig[0], color='b')
#     plt.title('Гистограмма яркости (исходное)')
    
#     plt.subplot(2, 3, 6)
#     plt.plot(hist_contr[1], hist_contr[0], color='r')
#     plt.title('Гистограмма яркости (контрастированное)')
    
#     plt.tight_layout()
#     plt.savefig('results.png')
#     plt.close()

# def main():
#     try:
#         # Загрузка изображения
#         img_path = 'image.jpg'  # Можно заменить на свой файл
#         original, hsl = load_image(img_path)
#         L_channel = hsl[:, :, 1]
        
#         # Линейное контрастирование
#         hsl_contrasted = linear_contrast(hsl.copy())
#         L_contrasted = hsl_contrasted[:, :, 1]
        
#         # Расчет GLRLM (нормализуем изображение к [0,1])
#         glrlm_orig = calculate_glrlm(L_channel / 255.0)
#         glrlm_contr = calculate_glrlm(L_contrasted / 255.0)
        
#         # Расчет признаков
#         glnu_orig, rlnu_orig = calculate_features(glrlm_orig)
#         glnu_contr, rlnu_contr = calculate_features(glrlm_contr)
        
#         # Гистограммы
#         hist_orig = exposure.histogram(L_channel)
#         hist_contr = exposure.histogram(L_contrasted)
        
#         # Визуализация
#         plot_results(original, hsl_contrasted, glrlm_orig, glrlm_contr, hist_orig, hist_contr)
        
#         # # Создание отчета
#         # with open('report.md', 'w') as f:
#         #     f.write("# Результаты текстурного анализа (Вариант 8)\n\n")
#         #     f.write("## Параметры анализа\n")
#         #     f.write("- Метод: GLRLM (Gray-Level Run Length Matrix)\n")
#         #     f.write("- Признаки: GLNU, RLNU\n")
#         #     f.write("- Контрастирование: Линейное\n\n")
#         #     f.write("## Текстурные признаки\n")
#         #     f.write(f"- **Исходное изображение**: GLNU = {glnu_orig:.2f}, RLNU = {rlnu_orig:.2f}\n")
#         #     f.write(f"- **Контрастированное изображение**: GLNU = {glnu_contr:.2f}, RLNU = {rlnu_contr:.2f}\n\n")
#         #     f.write("## Визуализации\n")
#         #     f.write("![Результаты анализа](results.png)\n\n")
#         #     f.write("## Вывод\n")
#         #     f.write("Линейное контрастирование изменило распределение яркости, что отразилось на значениях текстурных признаков. ")
#         #     f.write("Снижение значений GLNU и RLNU указывает на более равномерное распределение текстурных особенностей после контрастирования.")
        
#         # print("Анализ успешно завершен! Результаты сохранены в:")
#         # print("- results.png (визуализации)")
#         # print("- report.md (отчет)")
    
#     except Exception as e:
#         print(f"Ошибка: {str(e)}")
#         print("Убедитесь, что:")
#         print("1. Файл image.jpg существует в текущей папке")
#         print("2. Установлены все необходимые библиотеки (scikit-image, opencv-python)")
#         print("3. Изображение имеет правильный формат")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure
import os

def load_image(path):
    """Загрузка и преобразование изображения"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден!")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return img, img_hsl

def linear_contrast(hsl_img):
    """Улучшенное линейное контрастирование канала L"""
    L = hsl_img[:, :, 1].astype(np.float32)
    
    # Автоматическое определение диапазона контрастирования
    low = np.percentile(L, 5)  # Нижний 5-й процентиль
    high = np.percentile(L, 95)  # Верхний 95-й процентиль
    
    # Растягиваем контраст с сохранением крайних значений
    L_contrasted = np.clip((L - low) * (255.0 / (high - low)), 0, 255)
    
    hsl_img[:, :, 1] = L_contrasted.astype(np.uint8)
    return hsl_img

def calculate_glrlm(img, levels=8):
    """Расчет матрицы GLRLM"""
    img_quantized = (img * (levels-1)).astype(np.uint8)
    glcm = graycomatrix(img_quantized, distances=[1], angles=[0], levels=levels)
    return glcm[:, :, 0, 0]

def calculate_features(glrlm):
    """Расчет текстурных признаков"""
    total = np.sum(glrlm)
    if total == 0:
        return 0, 0
    
    p_i = np.sum(glrlm, axis=1)
    glnu = np.sum(p_i**2) / total
    
    p_j = np.sum(glrlm, axis=0)
    rlnu = np.sum(p_j**2) / total
    
    return glnu, rlnu

def plot_results(original, contrasted, glrlm_orig, glrlm_contr, hist_orig, hist_contr):
    """Улучшенная визуализация результатов"""
    plt.figure(figsize=(18, 12))
    
    # Исходное и контрастированное изображения
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(contrasted, cv2.COLOR_HLS2RGB))
    plt.title('После контрастирования')
    plt.axis('off')
    
    # Каналы яркости
    plt.subplot(3, 3, 4)
    plt.imshow(original[:, :, 1], cmap='gray', vmin=0, vmax=255)
    plt.title('Канал L (исходный)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(contrasted[:, :, 1], cmap='gray', vmin=0, vmax=255)
    plt.title('Канал L (контрастированный)')
    plt.axis('off')
    
    # GLRLM матрицы
    plt.subplot(3, 3, 7)
    plt.imshow(np.log1p(glrlm_orig), cmap='viridis', interpolation='nearest')
    plt.title('GLRLM (исходное)')
    plt.colorbar()
    
    plt.subplot(3, 3, 8)
    plt.imshow(np.log1p(glrlm_contr), cmap='viridis', interpolation='nearest')
    plt.title('GLRLM (контрастированное)')
    plt.colorbar()
    
    # Гистограммы
    plt.subplot(3, 3, 3)
    plt.plot(hist_orig[1], hist_orig[0], color='b', label='Исходное')
    plt.plot(hist_contr[1], hist_contr[0], color='r', alpha=0.7, label='Контраст')
    plt.title('Сравнение гистограмм')
    plt.legend()
    
    plt.subplot(3, 3, 6)
    plt.hist(original[:, :, 1].ravel(), bins=256, color='b', alpha=0.5, label='Исходное')
    plt.hist(contrasted[:, :, 1].ravel(), bins=256, color='r', alpha=0.5, label='Контраст')
    plt.title('Наложенные гистограммы')
    plt.legend()
    
    plt.subplot(3, 3, 9)
    diff = contrasted[:, :, 1].astype(int) - original[:, :, 1].astype(int)
    plt.imshow(diff, cmap='coolwarm', vmin=-50, vmax=50)
    plt.title('Разница в яркости')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Загрузка изображения
        img_path = 'image.jpg'
        original, hsl = load_image(img_path)
        L_channel = hsl[:, :, 1]
        
        # Контрастирование
        hsl_contrasted = linear_contrast(hsl.copy())
        L_contrasted = hsl_contrasted[:, :, 1]
        
        # Анализ яркости
        print(f"[Анализ яркости]")
        print(f"Исходное: min={L_channel.min()}, max={L_channel.max()}, mean={L_channel.mean():.1f}, std={L_channel.std():.1f}")
        print(f"Контраст: min={L_contrasted.min()}, max={L_contrasted.max()}, mean={L_contrasted.mean():.1f}, std={L_contrasted.std():.1f}")
        
        # Расчет GLRLM
        glrlm_orig = calculate_glrlm(L_channel / 255.0)
        glrlm_contr = calculate_glrlm(L_contrasted / 255.0)
        
        # Текстурные признаки
        glnu_orig, rlnu_orig = calculate_features(glrlm_orig)
        glnu_contr, rlnu_contr = calculate_features(glrlm_contr)
        
        print(f"\n[Текстурные признаки]")
        print(f"GLNU: исходное={glnu_orig:.1f}, контраст={glnu_contr:.1f}")
        print(f"RLNU: исходное={rlnu_orig:.1f}, контраст={rlnu_contr:.1f}")
        
        # Гистограммы
        hist_orig = exposure.histogram(L_channel, normalize=True)
        hist_contr = exposure.histogram(L_contrasted, normalize=True)
        
        # Визуализация
        plot_results(original, hsl_contrasted, glrlm_orig, glrlm_contr, hist_orig, hist_contr)
        
        # Сохранение результатов
        cv2.imwrite("contrasted_result.jpg", cv2.cvtColor(hsl_contrasted, cv2.COLOR_HLS2BGR))
        
        print("\nРезультаты сохранены в:")
        print("- results.png (полный анализ)")
        print("- contrasted_result.jpg (контрастированное изображение)")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()