import cv2
import numpy as np
import os

# Настройки для точного соответствия Calibri
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
size = (60, 100)  # Уменьшенный размер для точного соответствия

os.makedirs('templates', exist_ok=True)

# Специальные положения для букв
y_positions = {
    'g': 75, 'j': 75, 'p': 75, 'q': 75, 'y': 75,
    'default': 70
}

for char in 'abcdefghijklmnopqrstuvwxyz':
    img = np.zeros(size, dtype=np.uint8) + 255  # Белый фон
    y_pos = y_positions.get(char, y_positions['default'])
    
    # Черный текст на белом фоне
    cv2.putText(img, char, (20, y_pos), font, font_scale, 0, thickness)
    
    # Сохраняем как есть (черный текст на белом)
    cv2.imwrite(f'templates/{char}.png', img)

# import cv2
# import numpy as np
# import os

# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# thickness = 2
# size = (60, 100)

# os.makedirs('templates', exist_ok=True)

# y_positions = {
#     'g': 75, 'j': 75, 'p': 75, 'q': 75, 'y': 75,
#     'default': 70
# }

# for char in 'abcdefghijklmnopqrstuvwxyz':
#     img = np.zeros(size, dtype=np.uint8) + 255  # Белый фон
#     y_pos = y_positions.get(char, y_positions['default'])
    
#     # Черный текст на белом фоне
#     cv2.putText(img, char, (20, y_pos), font, font_scale, 0, thickness)
    
#     # Сохраняем как BMP
#     cv2.imwrite(f'templates/{char}.bmp', img)
