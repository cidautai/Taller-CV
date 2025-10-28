from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

def image_to_ascii_art(img, output_path = None, cols=100, scale=1.25, font_size=10):
    """
    Convert an image to ASCII art and save as PNG
    
    Parameters:
    - img: numpy array of the image
    - output_path: path for output PNG
    - cols: number of ASCII characters horizontally
    - scale: aspect ratio scale factor
    - font_size: font size for ASCII characters
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Extended ASCII characters from dark to light (more balanced set)
    ascii_chars = ['@', '%', '#', '*', '+', '=', '-', ':', '.', ' ']
    # Alternative sets you can try:
    # ascii_chars = ['$', '@', 'B', '%', '8', '&', 'W', 'M', '#', '*', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'w', 'm', 'Z', 'O', '0', 'Q', 'L', 'C', 'J', 'U', 'Y', 'X', 'z', 'c', 'v', 'u', 'n', 'x', 'r', 'j', 'f', 't', '/', '\\', '|', '(', ')', '1', '{', '}', '[', ']', '?', '-', '_', '+', '~', '<', '>', 'i', '!', 'l', 'I', ';', ':', ',', '"', '^', '`', "'", '.', ' ']
    # ascii_chars = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']
    
    print(np.shape(img))
    # Calculate dimensions
    width, height = np.shape(img)
    cell_width = width / cols
    cell_height = scale * cell_width
    rows = int(height / cell_height)
    
    # Resize image to match ASCII dimensions
    if cols > width or rows > height:
        print("Warning: ASCII dimensions larger than image, quality may suffer")
    
    img = cv2.resize(img, (cols, rows), cv2.INTER_LANCZOS4)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply contrast adjustment
    img_array = adjust_contrast(img_array, 1.2)
    
    # Create output image with white background
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
        # Get actual character dimensions
        try:
            char_width = font.getbbox("A")[2]  # Width of character 'A'
            char_height = font.getbbox("A")[3]  # Height of character 'A'
        except:
            # Fallback for older PIL versions
            char_width, char_height = font.getsize("A")
    except:
        print("Using default font measurements")
        char_width, char_height = 8, 16
    
    # Calculate output image size
    output_width = cols * char_width
    output_height = rows * char_height
    
    # Create output image
    output_img = Image.new('RGB', (output_width, output_height), 'white')
    draw = ImageDraw.Draw(output_img)
    
    # Convert pixels to ASCII and draw on output image
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
    
    # Save the output image
    if output_path:
        output_img.save(output_path, 'PNG')
        print(f"ASCII art saved to: {output_path}")
        print(f"Output dimensions: {output_width}x{output_height}")
        print(f"Used {len(ascii_chars)} ASCII characters")
    return output_img


def adjust_contrast(image_array, factor):
    """
    Adjust contrast of the image array
    factor > 1 increases contrast, < 1 decreases contrast
    """
    mean = np.mean(image_array)
    return np.clip((image_array - mean) * factor + mean, 0, 255).astype(np.uint8)

def map_intensity_to_ascii(intensity, num_chars):
    """
    Map grayscale intensity to ASCII character index with better distribution
    Uses non-linear mapping for better visual results
    """
    # Normalize to 0-1 range
    normalized = intensity / 255.0
    
    # Apply gamma correction for better perceptual mapping
    gamma = 1.5
    corrected = normalized ** gamma
    
    # Map to ASCII index
    ascii_index = int(corrected * (num_chars - 1))
    
    # Ensure within bounds
    return min(ascii_index, num_chars - 1)


