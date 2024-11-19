import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Función para crear un filtro pasa banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Aplicar filtro pasa banda
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Eliminar silencios del audio
def remove_silence(audio, top_db=25):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

# Dividir audio en segmentos iguales
def segment_audio(audio, num_segments=10):
    segment_length = len(audio) // num_segments
    segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
    return segments

# Calcular ZCR para un segmento con un n_fft ajustable
def extract_zcr(segment, n_fft=512):
    if len(segment) < n_fft:
        n_fft = len(segment)  # Ajustar n_fft al tamaño del segmento si es más pequeño
    zcr = librosa.feature.zero_crossing_rate(segment, frame_length=n_fft, hop_length=n_fft//2)[0]
    return np.mean(zcr)

# Calcular MFCC para un segmento, con selección de coeficientes específicos y n_fft ajustable
def extract_mfcc(segment, sr, selected_coeffs=[1, 4, 5], n_fft=512):
    # Ajustar n_fft si el segmento es muy corto
    if len(segment) < n_fft:
        n_fft = len(segment)
    
    # Ajustar n_mels y fmax en función de la tasa de muestreo
    n_mels = min(40, n_fft // 2)
    fmax = sr // 2 if sr // 2 < 8000 else 8000
    
    # Calcular MFCC con ajustes dinámicos
    mfcc = librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=max(selected_coeffs), n_fft=n_fft, hop_length=n_fft // 2, n_mels=n_mels, fmax=fmax
    )
    
    # Verificar si hay suficientes coeficientes disponibles
    if mfcc.shape[0] < max(selected_coeffs) + 1:
        selected_coeffs = [c for c in selected_coeffs if c < mfcc.shape[0]]
    
    # Si no hay suficientes coeficientes, omitir el segmento
    if len(selected_coeffs) == 0:
        return np.zeros(len(selected_coeffs))

    selected_mfcc = mfcc[selected_coeffs, :]
    return np.mean(selected_mfcc, axis=1)


# Calcular amplitud de un segmento
def extract_amplitude(segment):
    amplitude = np.max(np.abs(segment))
    return amplitude

# Ajustar la extracción de características para omitir segmentos muy cortos
def extract_features(segments, sr, filename, selected_coeffs=[1, 4, 5]):
    features = []
    for i, segment in enumerate(segments):
        if len(segment) < 512:  # Saltar segmentos demasiado cortos
            print(f"Segmento {i+1} del archivo {filename} es demasiado corto para el análisis.")
            continue
        
        zcr = extract_zcr(segment)
        mfcc = extract_mfcc(segment, sr, selected_coeffs)
        amplitude = extract_amplitude(segment)
        features.append({
            'Segmento': i + 1,
            'Archivo': filename,
            'ZCR': zcr,
            'MFCC': mfcc,
            'Amplitud': amplitude
        })
    return features

# Preparar los datos para PCA con etiquetas
def prepare_data_for_pca(features):
    data = []
    labels = []
    categories = []
    for feature in features:
        data.append([feature['ZCR']] + list(feature['MFCC']) + [feature['Amplitud']])
        labels.append(f"{feature['Archivo']}_seg{feature['Segmento']}")

        # Asignar categoría basado en el nombre del archivo
        if 'berenjena' in feature['Archivo'].lower():
            categories.append('berenjena')
        elif 'camote' in feature['Archivo'].lower():
            categories.append('camote')
        elif 'zanahoria' in feature['Archivo'].lower():
            categories.append('zanahoria')
        elif 'papa' in feature['Archivo'].lower():
            categories.append('papa')
        else:
            categories.append('desconocido')
    
    return np.array(data), labels, categories

# Aplicar PCA y graficar los resultados por categoría
def apply_pca_and_plot_by_category(features, n_components=2, threshold=0.2):
    data, labels, categories = prepare_data_for_pca(features)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)

    # Crear un diccionario para separar los datos por categoría
    category_dict = {
        'berenjena': [],
        'camote': [],
        'zanahoria': [],
        'papa': []
    }
    
    # Separar los puntos según la categoría
    for i, category in enumerate(categories):
        if category in category_dict:
            category_dict[category].append((pca_result[i, 0], pca_result[i, 1], labels[i]))

    # Generar gráficos para cada categoría
    for category, points in category_dict.items():
        if points:
            plt.figure(figsize=(10, 8))
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            labels = [p[2] for p in points]
            
            plt.scatter(x_vals, y_vals, c='blue', marker='o')
            plt.title(f'PCA para la categoría: {category}')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.grid()

            # Agregar etiquetas a los puntos
            for x, y, label in zip(x_vals, y_vals, labels):
                plt.annotate(label, (x, y), fontsize=8)

            plt.show()

# Procesar y extraer características de un archivo de audio
def process_segment_and_extract_features(file_path, lowcut=50, highcut=6000.0, top_db=20, num_segments=10, save_segments=False, selected_coeffs=[1, 4, 5]):
    audio, sr = librosa.load(file_path, sr=None)
    filtered_audio = bandpass_filter(audio, lowcut, highcut, sr)
    processed_audio = remove_silence(filtered_audio, top_db)
    segments = segment_audio(processed_audio, num_segments)

    if save_segments:
        segment_folder = os.path.join(os.path.dirname(file_path), 'segmentos')
        os.makedirs(segment_folder, exist_ok=True)
        for i, segment in enumerate(segments):
            segment_path = os.path.join(segment_folder, f'segment_{i+1}_{os.path.basename(file_path)}')
            sf.write(segment_path, segment, sr)

    features = extract_features(segments, sr, os.path.basename(file_path), selected_coeffs)
    return features

# Procesar todos los archivos y extraer características globalmente
def process_all_files(folder_path, lowcut=300.0, highcut=5000.0, top_db=20, num_segments=10, save_segments=False, selected_coeffs=[1, 4, 5]):
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            try:
                features = process_segment_and_extract_features(
                    audio_path, lowcut, highcut, top_db, num_segments, save_segments, selected_coeffs
                )
                all_features.extend(features)
            except Exception as e:
                print(f"Error procesando {audio_path}: {e}")
    return all_features

# Procesar todos los archivos y generar gráficos por categoría
folder_path = 'base_datos_voz/'
all_features = process_all_files(folder_path, save_segments=True)
apply_pca_and_plot_by_category(all_features)