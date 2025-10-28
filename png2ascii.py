from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

def image_to_ascii_art(img, output_path = None, cols=100, scale=1.25, font_size=10):
    """
    Convierte una imagen a ASCII y la guarda como PNG

    Parámetros:
    - img: array numpy de la imagen
    - output_path: ruta para el PNG de salida
    - cols: número de caracteres ASCII horizontalmente
    - scale: factor de escala para la relación de aspecto
    - font_size: tamaño de fuente para los caracteres ASCII
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Extended ASCII characters from dark to light (more balanced set)
    ascii_chars = ['@', '%', '#', '*', '+', '=', '-', ':', '.', ' ']
    # Alternative sets you can try:
    # ascii_chars = ['$', '@', 'B', '%', '8', '&', 'W', 'M', '#', '*', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'w', 'm', 'Z', 'O', '0', 'Q', 'L', 'C', 'J', 'U', 'Y', 'X', 'z', 'c', 'v', 'u', 'n', 'x', 'r', 'j', 'f', 't', '/', '\\', '|', '(', ')', '1', '{', '}', '[', ']', '?', '-', '_', '+', '~', '<', '>', 'i', '!', 'l', 'I', ';', ':', ',', '"', '^', '`', "'", '.', ' ']
    # ascii_chars = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']
    
    print(np.shape(img))
    # Calcular dimensiones
    width, height = np.shape(img)
    cell_width = width / cols
    cell_height = scale * cell_width
    rows = int(height / cell_height)
    
    # Resize image to match ASCII dimensions
    if cols > width or rows > height:
        print("Imagen demasiado pequeña para los parámetros dados.")
    
    img = cv2.resize(img, (cols, rows), cv2.INTER_LANCZOS4)
    
    # Convertimos la imagen a un array numpy para facilitar el acceso a los píxeles
    img_array = np.array(img)
    
    # Ajustamos el contraste para mejorar la visibilidad
    img_array = adjust_contrast(img_array, 1.2)
    
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
        # Obtener tamaño de un carácter
        try:
            char_width = font.getbbox("A")[2]  # Ancho del carácter 'A'
            char_height = font.getbbox("A")[3]  # Alto del carácter 'A'
        except:
            # Usar esto en caso de una versión antigua de PIL
            char_width, char_height = font.getsize("A")
    except:
        print("Usando medidas de fuente predeterminadas")
        char_width, char_height = 8, 16
    
    # Calculamos tamaño de imagen de salida
    output_width = cols * char_width
    output_height = rows * char_height

    # Creamos la imagen de salida
    output_img = Image.new('RGB', (output_width, output_height), 'white')
    draw = ImageDraw.Draw(output_img)
    
    # Convertimos los pixeles a ASCII y escribimos estos caracteres en la imagen de salida
    for y in range(rows):
        for x in range(cols):
            pixel_value = img_array[y, x]
            # Map grayscale value (0-255) to ASCII character index
            ascii_index = map_intensity_to_ascii(pixel_value, len(ascii_chars))
            ascii_char = ascii_chars[ascii_index]
            
            # Draw the ASCII character
            draw.text((x * char_width, y * char_height), 
                     ascii_char, 
                     font=font, 
                     fill='black')
    
    # Guardamos la imagen de salida
    if output_path:
        output_img.save(output_path, 'PNG')
        print(f"ASCII art saved to: {output_path}")
        print(f"Output dimensions: {output_width}x{output_height}")
        print(f"Used {len(ascii_chars)} ASCII characters")
    return output_img


def adjust_contrast(image_array, factor):
    """
    Ajusta el contraste del array de imagen
    factor > 1 aumenta el contraste, < 1 lo disminuye
    """
    mean = np.mean(image_array)
    return np.clip((image_array - mean) * factor + mean, 0, 255).astype(np.uint8)

def map_intensity_to_ascii(intensity, num_chars):
    """
    Mapea la intensidad en escala de grises al índice de carácter ASCII con mejor distribución
    Utiliza un mapeo no lineal para mejores resultados visuales
    """
    # Normaliza al rango 0-1
    normalized = intensity / 255.0
    
    # Aplica corrección gamma para un mapeo perceptual mejor
    gamma = 1.5
    corrected = normalized ** gamma
    
    # Mapea al índice ASCII
    ascii_index = int(corrected * (num_chars - 1))
    
    # Asegura que esté dentro de los límites
    return min(ascii_index, num_chars - 1)


