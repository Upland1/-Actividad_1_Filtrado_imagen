import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_plate_image(image_path):
    """
    Procesa una imagen de una placa para generar una versión subsampleada
    en escala de grises con filtro Gaussiano aplicado a una mascara invertida

    """
    # Leer la imagen
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Convertir a RGB para visualización y obtener dimensiones de la imagem
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Crear mascara e invertirla
    dark_mask = gray_image < 30
    inverted_mask = (dark_mask.astype(np.uint8)) * 255

    # Aplicar desenfoque gaussiano
    blurred_mask = cv2.GaussianBlur(inverted_mask, (15, 15), 0)

    # Subsampled con interpolacion de area
    subsampled = cv2.resize(
        blurred_mask,
        (width // 8, height // 8),
        interpolation=cv2.INTER_AREA
    )

    return subsampled