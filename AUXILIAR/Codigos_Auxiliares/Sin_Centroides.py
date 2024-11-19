import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import local_binary_pattern


def extract_hue_feature(image):
    """Extrae el valor promedio del canal H (Hue) en el espacio HSV."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    return np.mean(hue_channel)


def extract_texture_feature(image):
    """Calcula el LBP (Local Binary Pattern) para medir la textura."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Parámetros de LBP: P = 8 puntos, R = 1 radio, método 'uniform'
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    # Retornamos el valor promedio del patrón LBP
    return np.mean(lbp)

def extract_second_hu_moment(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments[1]


def extract_circularity(contour):
    """Calcula la circularidad del contorno."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

def preprocess_and_extract_features(image_path, show_plots=False):
    """Preprocesa la imagen, detecta el contorno más grande y extrae características."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (800, 800))
    blurred_image = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
    # Convertir al espacio de color HSV
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)
    hsv_image = cv2.convertScaleAbs(hsv_image, alpha=1.2, beta=30)
    
    gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)
    _, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detectar contornos
    contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        hu_moment = extract_second_hu_moment(largest_contour)
        circularity = extract_circularity(largest_contour)
        texture_feature = extract_texture_feature(image_resized)
        hue_feature = extract_hue_feature(image_resized)
        
        if show_plots:
            contour_image = image_resized.copy()
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Imagen Original Redimensionada")
            
            axes[1].imshow(binary_otsu, cmap='gray')
            axes[1].set_title("Binarización Otsu")
            
            axes[2].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Contorno Detectado")
            
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        
        return hu_moment, circularity, texture_feature, hue_feature
    return None, None, None, None


def load_images_and_extract_features(folder_path, show_plots=False):
    """Carga las imágenes desde la carpeta y extrae características."""
    data = {
        'filename': [], 
        'label': [], 
        'second_hu_moment': [], 
        'circularity': [], 
        'texture': [], 
        'hue': []
    }
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        hu_moment, circularity, texture_feature, hue_feature = preprocess_and_extract_features(image_path, show_plots=show_plots)
        if hu_moment is not None:
            label = image_file.split('_')[0]
            data['filename'].append(image_file)
            data['label'].append(label)
            data['second_hu_moment'].append(hu_moment)
            data['circularity'].append(circularity)
            data['texture'].append(texture_feature)
            data['hue'].append(hue_feature)
    
    return pd.DataFrame(data)


def plot_pca_3d(dataframe):
    """Realiza el análisis PCA en 3D y visualiza los resultados."""
    pca = PCA(n_components=3)
    features = dataframe[['second_hu_moment', 'circularity', 'texture', 'hue']].values
    transformed = pca.fit_transform(features)
    
    dataframe['PCA1'] = transformed[:, 0]
    dataframe['PCA2'] = transformed[:, 1]
    dataframe['PCA3'] = transformed[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for label in dataframe['label'].unique():
        subset = dataframe[dataframe['label'] == label]
        ax.scatter(subset['PCA1'], subset['PCA2'], subset['PCA3'], label=label)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D PCA Analysis with Hue')
    ax.legend()
    plt.show()


# Cargar imágenes, extraer características y realizar análisis PCA
folder_path = 'base_datos_img'
df = load_images_and_extract_features(folder_path, show_plots=False)
if not df.empty:
    plot_pca_3d(df)
