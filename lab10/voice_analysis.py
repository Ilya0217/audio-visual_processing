import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import soundfile as sf
import os

def load_audio(file_path):
    """Загрузка и нормализация аудио с проверкой"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if len(y) == 0:
            raise ValueError("Аудиофайл пустой")
        return y / (np.max(np.abs(y)) + 1e-10), sr
    except Exception as e:
        raise ValueError(f"Ошибка загрузки {file_path}: {str(e)}")

def plot_spectrogram(y, sr, title, filename):
    """Построение спектрограммы с обработкой ошибок"""
    try:
        plt.figure(figsize=(12, 6))
        nperseg = min(2048, len(y)//4)
        noverlap = int(nperseg * 0.75)
        
        f, t, Sxx = signal.spectrogram(y, sr, window='hann',
                                     nperseg=nperseg, noverlap=noverlap,
                                     scaling='spectrum', mode='magnitude')
        
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-10), 
                      shading='gouraud', vmin=-60, vmax=60)
        plt.yscale('symlog', linthresh=100)
        plt.ylim(50, sr//2)
        plt.ylabel('Частота [Гц]')
        plt.xlabel('Время [с]')
        plt.title(title)
        plt.colorbar(label='Уровень [дБ]')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"Ошибка построения спектрограммы: {str(e)}")
        return False

def find_frequency_range(y, sr):
    """Нахождение частотного диапазона с проверкой"""
    try:
        f, Pxx = signal.welch(y, sr, nperseg=min(1024, len(y)//2))
        peaks = signal.find_peaks(Pxx, height=np.max(Pxx)/10, distance=10)[0]
        
        if len(peaks) == 0:
            return (0, 0)
        
        freqs = f[peaks]
        valid_freqs = freqs[(freqs >= 50) & (freqs <= 5000)]
        
        if len(valid_freqs) == 0:
            return (0, 0)
        
        return (np.min(valid_freqs), np.max(valid_freqs))
    except:
        return (0, 0)

def find_timbre_frequency(y, sr):
    """Нахождение основной частоты с проверкой"""
    try:
        # Метод автокорреляции
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Ищем пики, исключая первый (нулевой сдвиг)
        peaks = signal.find_peaks(autocorr, distance=sr//200)[0]
        if len(peaks) < 2:
            return 0
        
        fundamental = sr / peaks[1]
        return fundamental if 50 <= fundamental <= 1000 else 0
    except:
        return 0

def find_formants(y, sr, n_formants=3):
    """Надежный поиск формант"""
    try:
        # Используем метод LPC для анализа формант
        n_coeff = int(2 + sr / 1000)
        a = librosa.lpc(y, order=n_coeff)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]  # Берем верхнюю полуплоскость
        
        # Вычисляем частоты формант
        angs = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angs * (sr / (2 * np.pi))
        
        # Берем первые n формант в речевом диапазоне
        valid_freqs = sorted([f for f in freqs if 200 < f < 5000], reverse=True)
        return valid_freqs[:n_formants]
    except:
        return [0, 0, 0]

def analyze_voice(file_prefix, description):
    """Полный анализ с обработкой ошибок"""
    try:
        y, sr = load_audio(f"{file_prefix}.wav")
        
        # Спектрограмма
        spec_success = plot_spectrogram(y, sr, 
                                      f'Спектрограмма: {description}',
                                      f'{file_prefix}_spectrogram.png')
        
        # Анализ характеристик
        fmin, fmax = find_frequency_range(y, sr)
        timbre_freq = find_timbre_frequency(y, sr)
        formants = find_formants(y, sr)
        
        # Заполняем недостающие форманты нулями
        while len(formants) < 3:
            formants.append(0)
        
        return {
            'description': description,
            'fmin': fmin,
            'fmax': fmax,
            'timbre': timbre_freq,
            'formants': formants[:3],  # Всегда 3 форманты
            'duration': len(y)/sr,
            'spectrogram': f'{file_prefix}_spectrogram.png' if spec_success else None
        }
    except Exception as e:
        print(f"Ошибка анализа {file_prefix}: {str(e)}")
        return None

def create_report(results):
    """Создание отчета с проверкой данных"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write("# Лабораторная работа №10: Обработка голоса\n\n")
        
        f.write("## Результаты анализа\n\n")
        
        for res in results:
            if not res:
                continue
                
            f.write(f"### {res['description']}\n")
            f.write(f"- Длительность: {res['duration']:.2f} с\n")
            f.write(f"- Частотный диапазон: {res['fmin']:.1f} - {res['fmax']:.1f} Гц\n")
            f.write(f"- Основная частота: {res['timbre']:.1f} Гц\n")
            f.write("- Топ-3 форманты:\n")
            for i, freq in enumerate(res['formants']):
                f.write(f"  {i+1}. {freq:.1f} Гц\n")
            
            if res['spectrogram']:
                f.write(f"![Спектрограмма]({res['spectrogram']})\n")
            else:
                f.write("(Спектрограмма не доступна)\n")
            f.write("\n")
        
        f.write("## Выводы\n")
        f.write("1. Разные звуки имеют характерные формантные структуры\n")
        f.write("2. Основная частота соответствует высоте голоса\n")
        f.write("3. Спектрограммы показывают распределение энергии по частотам\n")

def main():
    print("=== Анализ голосовых записей ===")
    
    # Список файлов для анализа
    files_to_analyze = [
        ('a', 'Звук А'),
        ('i', 'Звук И'), 
        ('gav', 'Имитация животного')
    ]
    
    results = []
    for file_prefix, description in files_to_analyze:
        print(f"\nАнализируем {description}...")
        result = analyze_voice(file_prefix, description)
        if result:
            results.append(result)
    
    # Создание отчета
    create_report(results)
    
    # Вывод информации о созданных файлах
    print("\nСозданы файлы:")
    for res in results:
        if res and res['spectrogram']:
            print(f"- {res['spectrogram']}")
    print("- README.md")

if __name__ == "__main__":
    main()