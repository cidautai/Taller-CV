import cv2
import numpy as np

def quick_quantize_video(input_path, output_path, colors=16):
    """
    Quantización de color rápida y sencilla
    """
    cap = cv2.VideoCapture(input_path)
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Quantización uniforme simple (método más rápido)
        shift = 8 - int(np.log2(colors))
        quantized = (frame >> shift) << shift
        
        out.write(quantized)
    
    cap.release()
    out.release()
    print(f"Vídeo cuantizado con {colors} colores!")

# Ejemplo de uso:
# quick_quantize_video('input.mp4', 'output.mp4', colors=8)

def quantize_image(image, k=16, method='kmeans'):
    """
    Quantización de color rápida para una sola imagen
    
    Args:
        image: Imagen de OpenCV (array de numpy) en formato BGR
        k: Número de colores (por defecto: 16)
        method: 'kmeans' (preciso) o 'uniform' (más rápido)
    
    Returns:
        Devuelve: Imagen quantizada como array de numpy
    """
    if method == 'kmeans':
        return kmeans_quantize(image, k)
    elif method == 'uniform':
        return uniform_quantize(image, k)
    elif method == 'median_cut':
        return median_cut_quantize(image, k)
    else:
        raise ValueError("Method must be 'kmeans' or 'uniform'")

def kmeans_quantize(image, k=16):
    """Cuantización de color con k-means para una sola imagen"""
    # Reorganizar la imagen a un array 2D de píxeles
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Aplicar k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir de nuevo a uint8 y reorganizar
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(image.shape)

def uniform_quantize(image, k=16):
    """Quantización uniforme rápida usando reducción de bits"""
    # Calcular los bits a conservar por canal
    bits = max(1, int(np.log2(k)))
    shift = 8 - bits
    return (image >> shift) << shift

def median_cut_quantize(image, k=16):
    """
    Método alternativo: quantización por corte por la mediana
    Suele producir buenos resultados con colores vibrantes
    """
    # Reorganizar a array de píxeles
    data = image.reshape((-1, 3))
    
    def median_cut(pixels, depth=0):
        if depth >= int(np.log2(k)) or len(pixels) == 0:
            # Devolver el color promedio
            return [np.mean(pixels, axis=0, dtype=np.uint8)]
        
        # Encontrar el canal de color con mayor rango
        ranges = np.ptp(pixels, axis=0)
        channel = np.argmax(ranges)
        
        # Ordenar por ese canal y dividir en la mediana
        pixels = pixels[pixels[:, channel].argsort()]
        median_idx = len(pixels) // 2
        
        # Procesar recursivamente ambas mitades
        return median_cut(pixels[:median_idx], depth + 1) + median_cut(pixels[median_idx:], depth + 1)
    
    # Obtener paleta
    palette = np.array(median_cut(data))
    
    # Asignar cada píxel al color de la paleta más cercano
    distances = np.linalg.norm(data[:, np.newaxis] - palette, axis=2)
    labels = np.argmin(distances, axis=1)
    quantized = palette[labels]
    
    return quantized.reshape(image.shape)
