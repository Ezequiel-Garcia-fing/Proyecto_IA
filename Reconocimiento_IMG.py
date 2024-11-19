import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

#Calcula la extensión de un contorno,proporción entre el área del contorno y el área del rectángulo que lo encierra.
def extract_extent(contour):
    """Calcula la extensión del contorno."""
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    rect_area = w * h
    if rect_area == 0:
        return 0
    return area / rect_area

#Calcula la media del canal de tonalidad (Hue) en el espacio de color HSV.
def extract_hue_feature(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    return np.mean(hue_channel)

#Extrae una característica relacionada con la textura de la imagen utilizando el patrón binario local (LBP).
def extract_texture_feature(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    return np.mean(lbp)

#Calcula los primeros cuatro momentos invariantes de Hu a partir de un contorno.
def extract_hu_moments(contour):
    """Calcula los primeros cuatro momentos de Hu a partir del contorno."""
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Devolver los primeros cuatro momentos de Hu
    return hu_moments[:4]  # Retornamos Hu_1, Hu_2, Hu_3 y Hu_4

#Esta función calcula la circularidad de un contorno, que es una medida de cuán cerca está un contorno de ser un círculo perfecto.
#=1 circulo perfceto, menor a uno forma irregular o alargada
def extract_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

#preprocesamiento y extraccion de caracteristicas de una sola imagenes para su posterior clasficacion.
#preprocess_and_extract_features realiza el preprocesameinto y extraccion de imagenes (individuales)
# que load_images_and_extract_features_with_normalization le va pasando y a su vez esta ultima funcion organiza los datos que le llegan de preprocess_and_extract_features?
def preprocess_and_extract_features(image_path, show_plots=False):
    """Preprocesa la imagen, detecta el contorno más grande y extrae características."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (800, 800))
    blurred_image = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)
    hsv_image = cv2.convertScaleAbs(hsv_image, alpha=1.2, beta=30)
    
    gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)
    _, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extraer características
        hu_moments = extract_hu_moments(largest_contour)
        circularity = extract_circularity(largest_contour)
        texture_feature = extract_texture_feature(image_resized)
        hue_feature = extract_hue_feature(image_resized)
        extent = extract_extent(largest_contour)
        
        # Mostrar los pasos del preprocesamiento si show_plots es True
        if show_plots:
            contour_image = image_resized.copy()
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Imagen Redimensionada")
            
            axes[1].imshow(binary_otsu, cmap='gray')
            axes[1].set_title("Binarización Otsu")
            
            axes[2].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Contorno Detectado")
            
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        
        return hu_moments.tolist() + [circularity, texture_feature, hue_feature, extent]
    return [None] * 8

def load_images_and_extract_features_with_normalization(folder_path, save_csv=False):
    """
    Carga imágenes desde una carpeta, extrae sus características y las normaliza.
    """
    data = {
        'filename': [], 'label': [],
        'hu_1': [], 'hu_2': [], 'hu_3': [], 'hu_4': [],
        'circularity': [], 'texture': [], 'hue': [], 'extent': []
    }
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        features = preprocess_and_extract_features(image_path)
        if None not in features:
            label = image_file.split('_')[0]
            data['filename'].append(image_file)
            data['label'].append(label)
            data['hu_1'].append(features[0])
            data['hu_2'].append(features[1])
            data['hu_3'].append(features[2])
            data['hu_4'].append(features[3])
            data['circularity'].append(features[4])
            data['texture'].append(features[5])
            data['hue'].append(features[6])
            data['extent'].append(features[7])
    
    df = pd.DataFrame(data)
    
    # Normalizar las características
    scaler = StandardScaler()
    feature_columns = ['hu_1', 'hu_2', 'hu_3', 'hu_4', 'circularity', 'texture', 'hue', 'extent']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    if save_csv:
        csv_path = 'caracteristicas_normalizadas.csv'
        df.to_csv(csv_path, index=False)
        print(f"Archivo CSV guardado en: {csv_path}")
    
    return df, scaler

# Visualización de la separación de clases y centroides:
def plot_pca_3d(dataframe, centroids=None):
    """Visualiza los datos y los centroides en un espacio PCA 3D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar los puntos de datos con sus etiquetas
    for label in dataframe['label'].unique():
        subset = dataframe[dataframe['label'] == label]
        ax.scatter(subset['PCA1'], subset['PCA2'], subset['PCA3'], label=label)
    
    # Graficar los centroides si están disponibles
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                   c='red', marker='X', s=200, label='Centroids')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D PCA Analysis with Centroids')
    ax.legend()
    plt.show()


#Se utiliza al inicio del algoritmo K-Means para calcular las coordenadas iniciales de los centroides.
# Calcula el promedio de los puntos de cada etiqueta en el espacio PCA y los utiliza como los centroides iniciales.
def initialize_centroids_pca_labels(dataframe, n_clusters=4):
    """
    Inicializa los centroides utilizando las etiquetas en el espacio PCA.
    """
    pca = PCA(n_components=3)
    features = dataframe[['hu_1', 'hu_2', 'hu_3', 'hu_4', 'circularity', 'texture', 'hue', 'extent']].values
    
    # Transformar las características al espacio PCA
    transformed_features = pca.fit_transform(features)
    
    # Añadir las componentes PCA al DataFrame para el gráfico
    dataframe['PCA1'] = transformed_features[:, 0]
    dataframe['PCA2'] = transformed_features[:, 1]
    dataframe['PCA3'] = transformed_features[:, 2]
    
    # Inicializar los centroides utilizando las etiquetas
    centroids = []
    for label in dataframe['label'].unique():
        subset = dataframe[dataframe['label'] == label]
        centroid = subset[['PCA1', 'PCA2', 'PCA3']].mean().values
        centroids.append(centroid)
    
    return np.array(centroids), transformed_features, pca


#Las etiquetas originales solo se usan al principio (si las tienes) para una inicialización guiada.
#Durante las iteraciones, el algoritmo es no supervisado, lo que significa que no tiene acceso a las etiquetas originales y trabaja únicamente con las etiquetas generadas basadas en las distancias a los centroides.
def adjust_centroids_with_labels(data, labels, n_clusters=4):
    """Recalcula los centroides basados en las asignaciones de clusters."""
    new_centroids = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
            new_centroids.append(new_centroid)
        else:
            # Si el cluster está vacío, mantenemos el centroide original
            new_centroids.append(np.zeros(data.shape[1]))
    return np.array(new_centroids)

#La asignación de puntos a clústeres en función de la distancia a los centroides.
#El control de iteraciones y la verificación de convergencia.
def kmeans_custom_pca_adjusted(data, initial_centroids, max_iterations=100):
    """K-means utilizando centroides inicializados en el espacio PCA."""
    centroids = initial_centroids
    for _ in range(max_iterations):
        # Calcular distancias a los centroides en el espacio PCA
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalcular los centroides usando adjust_centroids_with_labels
        new_centroids = adjust_centroids_with_labels(data, labels, n_clusters=len(centroids))

        # Verificar si los centroides han cambiado
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

#clasificar imágenes almacenadas en una carpeta de pruebas (pruebas_img) utilizando un modelo entrenado basado en PCA y centroides calculados por K-Means.
# las caracteristicas obtenidas se reducen al espacio pca porque las caracteristicas del entrenamiento tambien estan reducidas  
def recognize_images_pca(centroids, pca, scaler, label_map, show_plots=False):
    """
    Clasifica las imágenes en la carpeta 'pruebas_img' utilizando los ultimos centroides 
    """
    test_folder = 'pruebas_img'
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    
    for image_file in test_files:
        image_path = os.path.join(test_folder, image_file)
        features = preprocess_and_extract_features(image_path, show_plots=show_plots)
        
        if None not in features:
            # Convertir las características en un DataFrame con los mismos nombres de columnas que el entrenamiento
            feature_names = ['hu_1', 'hu_2', 'hu_3', 'hu_4', 'circularity', 'texture', 'hue', 'extent']
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # Normalizar utilizando el escalador entrenado
            normalized_features = scaler.transform(features_df)
            
            # Transformar las características normalizadas al espacio PCA
            transformed_features = pca.transform(normalized_features)
            
            # Calcular las distancias a los centroides en el espacio PCA
            distances = np.linalg.norm(centroids - transformed_features, axis=1)
            cluster = np.argmin(distances)
            predicted_label = label_map[cluster]
            
            #print(f"Imagen: {image_file}, Predicción: {predicted_label}")
            #print(f"Distancias a los centroides: {distances}")
            #print(f"Distancia mínima (centroide {cluster}): {distances[cluster]}")
            
            results.append({'Nombre de archivo': image_file, 'Prediccion': predicted_label})
    
    # Mostrar un resumen de las predicciones
    df_results = pd.DataFrame(results)
    print("\nTabla resumen de predicciones:")
    print(df_results)
    return df_results


# Cargar datos de entrenamiento y normalizarlos
df_train, scaler = load_images_and_extract_features_with_normalization('base_datos_img', save_csv=False)

# Reducir con PCA
pca = PCA(n_components=3)
transformed_features = pca.fit_transform(df_train[['hu_1', 'hu_2', 'hu_3', 'hu_4', 'circularity', 'texture', 'hue', 'extent']].values)

# Inicializar los centroides utilizando las etiquetas en el espacio PCA
initial_centroids, _, _ = initialize_centroids_pca_labels(df_train)

# Entrenar K-means utilizando los centroides inicializados en el espacio PCA
centroids, labels = kmeans_custom_pca_adjusted(transformed_features, initial_centroids)

# Crear un mapa de etiquetas basado en el entrenamiento
label_map = {i: df_train['label'][labels == i].mode()[0] for i in range(4)}

# Visualizar el gráfico PCA 3D con los datos y los centroides
plot_pca_3d(df_train, centroids=centroids)

# Clasificar las imágenes de la carpeta "pruebas_img"
recognize_images_pca(centroids=centroids, pca=pca, scaler=scaler, label_map=label_map, show_plots=True)


