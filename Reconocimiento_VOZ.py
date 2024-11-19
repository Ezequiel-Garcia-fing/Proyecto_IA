import os
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

# Función para crear un filtro pasa banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#Aplicar el filtro pasa banda a los datos de audio.
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

#Eliminar silencio en el audio.
def remove_silence(audio, top_db=20):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

#Normalizar el audio para que sus valores estén dentro del rango [−1,1].
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

#Dividir el audio en segmentos iguales.
def segment_audio(audio, num_segments=10):
    segment_length = len(audio) // num_segments
    return [audio[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]

#Para obtener un segmento de audio con la longitud deseada (solucion a problemas en la extraccion MFCC).
def pad_segment(segment, target_length=2048):
    if len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), 'constant')
    return segment

#Extraer los coeficientes cepstrales de frecuencia Mel (MFCC) seleccionados.
def extract_mfcc(segment, sr, selected_coeffs=[1, 4, 5], n_fft=512):
    segment = pad_segment(segment, n_fft)
    n_mels = 20
    fmax = min(8000, sr // 2)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=max(selected_coeffs) + 1,
                                n_fft=n_fft, hop_length=n_fft // 2,
                                n_mels=n_mels, fmax=fmax)
    selected_mfcc = mfcc[selected_coeffs, :] if mfcc.shape[0] >= max(selected_coeffs) + 1 else np.zeros(len(selected_coeffs))
    return np.mean(selected_mfcc, axis=1)

#Calcular la tasa de cruces por cero del segmento.
def extract_zcr(segment, n_fft=512):
    segment = pad_segment(segment, n_fft)
    return np.mean(librosa.feature.zero_crossing_rate(segment, frame_length=n_fft)[0])

#Obtener la amplitud máxima del segmento.
def extract_amplitude(segment):
    return np.max(np.abs(segment))

#Extraer el contraste espectral del segmento. Util para capturar la riqueza armónica y la variación tonal
def extract_spectral_contrast(segment, sr, n_fft=512):
    segment = pad_segment(segment, n_fft)
    spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_fft=n_fft)
    return np.mean(spectral_contrast, axis=1)

#graba audio en tiempo real desde el micrófono, ajusta su tasa de muestreo y lo guarda en un archivo .wav.
def record_and_save_audio(filename='grabacion_prueba.wav', duration=3, sample_rate=48000, target_rate=44100):
    """Graba audio en tiempo real y guarda como un archivo .wav con una tasa de muestreo ajustada."""
    print(f"Grabando durante {duration} segundos...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    # Convertir la tasa de muestreo a 44100 Hz
    audio_resampled = librosa.resample(audio_data.flatten(), orig_sr=sample_rate, target_sr=target_rate)
    write(filename, target_rate, audio_resampled)
    print(f"Grabación guardada como {filename}")

#calcula la distancia euclidiana
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# procesa cadaa archivo de la carpeta base de datos luego los segmenta y 
# luego llama a extract_features_for_file para Procesa todos los segmentos de un archivo, organizándolos
#Luego se llama a extract_features para obtener las caracteristicas de segmetnos en especifico
def process_all_files(folder_path):
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                filtered_audio = bandpass_filter(audio, 100, 6000, sr)
                processed_audio = remove_silence(filtered_audio)
                normalized_audio = normalize_audio(processed_audio)
                segments = segment_audio(normalized_audio)
                features = extract_features_for_file(segments, sr, filename)
                all_features.extend(features)
            except Exception as e:
                print(f"Error procesando {audio_path}: {e}")
    return all_features
#Procesa todos los segmentos de un archivo, organizándolos y añadiendo información adicional 
#como el nombre del archivo y el índice del segmento.
def extract_features_for_file(segments, sr, filename):
    features = []
    for i, segment in enumerate(segments):
        zcr = extract_zcr(segment)
        mfcc = extract_mfcc(segment, sr)
        amplitude = extract_amplitude(segment)
        spectral_contrast = extract_spectral_contrast(segment, sr)
        features.append({
            'Segmento': i + 1,
            'Archivo': filename,
            'ZCR': zcr,
            'MFCC': mfcc,
            'Amplitud': amplitude,
            'SpectralContrast': spectral_contrast
        })
    return features

#combina diferentes características del segemto para formar un vector de características completo
def extract_features(segment, sr):
    zcr = extract_zcr(segment)
    mfcc = extract_mfcc(segment, sr)
    amplitude = extract_amplitude(segment)
    spectral_contrast = extract_spectral_contrast(segment, sr)
    return np.hstack([zcr, mfcc, amplitude, spectral_contrast])

#diccionario especifica qué segmentos de cada palabra serán seleccionados para el análisis.
selected_segments_per_word = {
    'camote': [6],
    'berenjena': [6],
    'zanahoria': [1],
    'papa': [2]
}

#Esta función filtra y organiza los datos de características extraídas, 
# seleccionando solo los segmentos relevantes de acuerdo con el diccionario. Devuelve los datos en un formato adecuado para el análisis con PCA.
def prepare_data_for_pca(features):
    data, labels, categories = [], [], []
    for feature in features:
        category = feature['Archivo'].split('_')[0].lower()
        if category in selected_segments_per_word:
            if feature['Segmento'] in selected_segments_per_word[category]:
                data.append([feature['ZCR']] + list(feature['MFCC']) + [feature['Amplitud']] + list(feature['SpectralContrast']))
                labels.append(f"{feature['Archivo']}_seg{feature['Segmento']}")
                categories.append(category)
    return np.array(data), labels, categories

#útil para entender cómo se distribuyen los datos en el espacio PCA y si las clases son distinguibles.
def plot_additional_pca_3d(features, pca_result, labels, categories):
    colors = {'berenjena': 'yellow', 'camote': 'blue', 'zanahoria': 'purple', 'papa': 'green'}
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    for i, (x, y, z) in enumerate(pca_result):
        category = categories[i]
        color = colors.get(category, 'black')
        ax.scatter(x, y, z, color=color, label=category if category not in categories[:i] else "", alpha=0.8)
    ax.set_title("Visualización PCA de las características de voz")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_zlabel("Componente principal 3")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Clase de verdura", loc='best')
    plt.grid(True)
    plt.show()

#Filtra los segmentos relevantes según su categoría y número.
#Crea vectores de características combinadas (ZCR, MFCC, amplitud, contraste espectral).
#Organiza los vectores en un diccionario estructurado por clases, listo para usarse en un modelo KNN.
def prepare_knn_data(features):
    training_features = {'camote': [], 'berenjena': [], 'zanahoria': [], 'papa': []}
    for feature in features:
        category = feature['Archivo'].split('_')[0].lower()
        segment = feature['Segmento']
        if (category == 'camote' and segment == 6) or (category == 'berenjena' and segment == 6) or \
           (category == 'zanahoria' and segment == 1) or (category == 'papa' and segment == 2):
            feature_vector = np.hstack([feature['ZCR'], feature['MFCC'], feature['Amplitud'], feature['SpectralContrast']])
            training_features[category].append(feature_vector)
    return training_features

def KNN_casero(audio_path, training_features, target_rate=44100, umbral_zanahoria=70):
    audio, sr = librosa.load(audio_path, sr=target_rate)
    
    filtered_audio = bandpass_filter(audio, 100, 6000, sr)
    processed_audio = remove_silence(filtered_audio)
    normalized_audio = normalize_audio(processed_audio)
    segments = segment_audio(normalized_audio)

    distances = {}

    # Comparar segmento 1 con 'zanahoria'
    if len(segments) >= 1:
        feature_1 = extract_features(segments[0], sr)
        dist_zanahoria_seg1 = min([euclidean_distance(feature_1, train) for train in training_features['zanahoria']])
        distances['zanahoria_seg1'] = dist_zanahoria_seg1

    # Comparar segmento 2 con 'papa'
    if len(segments) >= 2:
        feature_2 = extract_features(segments[1], sr)
        distances['papa'] = min([euclidean_distance(feature_2, train) for train in training_features['papa']])

    # Comparar segmento 6 con 'camote' y 'berenjena'
    if len(segments) >= 6:
        feature_6 = extract_features(segments[5], sr)
        camote_distance = min([euclidean_distance(feature_6, train) for train in training_features['camote']])
        berenjena_distance = min([euclidean_distance(feature_6, train) for train in training_features['berenjena']])
        if camote_distance < berenjena_distance:
            distances['camote'] = camote_distance
        else:
            distances['berenjena'] = berenjena_distance

    # Si la distancia del segmento 1 de zanahoria es suficientemente baja, priorizarla
    if distances['zanahoria_seg1'] < umbral_zanahoria:
        print("\nLa palabra reconocida es: zanahoria")
        return 'zanahoria'

    # Si no se cumple el umbral, elegir según las demás distancias
    final_prediction = min(distances, key=distances.get)
    print(f"\nLa palabra reconocida es: {final_prediction}")
    return final_prediction

# Ejecutar el reconocimiento
folder_path = 'base_datos_voz/'
all_features = process_all_files(folder_path)
data, labels, categories = prepare_data_for_pca(all_features)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)
#plot_additional_pca_3d(all_features, pca_result, labels, categories)
training_features = prepare_knn_data(all_features)
record_and_save_audio('grabacion_prueba.wav')
KNN_casero('grabacion_prueba.wav', training_features)

