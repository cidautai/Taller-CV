import cv2
import numpy as np

def quick_quantize_video(input_path, output_path, colors=16):
    """
    Quick and simple color quantization
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Simple uniform quantization (fastest method)
        shift = 8 - int(np.log2(colors))
        quantized = (frame >> shift) << shift
        
        out.write(quantized)
    
    cap.release()
    out.release()
    print(f"Video quantized with {colors} colors!")

# Usage example:
# quick_quantize_video('input.mp4', 'output.mp4', colors=8)

def quantize_image(image, k=16, method='kmeans'):
    """
    Fast color quantization for a single image
    
    Args:
        image: OpenCV image (numpy array) in BGR format
        k: Number of colors (default: 16)
        method: 'kmeans' (accurate) or 'uniform' (fastest)
    
    Returns:
        Quantized image as numpy array
    """
    if method == 'kmeans':
        return kmeans_quantize(image, k)
    elif method == 'uniform':
        return uniform_quantize(image, k)
    else:
        raise ValueError("Method must be 'kmeans' or 'uniform'")

def kmeans_quantize(image, k=16):
    """K-means color quantization for single image"""
    # Reshape image to 2D array of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(image.shape)

def uniform_quantize(image, k=16):
    """Fast uniform quantization using bit reduction"""
    # Calculate bits to keep per channel
    bits = max(1, int(np.log2(k)))
    shift = 8 - bits
    return (image >> shift) << shift

def median_cut_quantize(image, k=16):
    """
    Alternative method: Median cut quantization
    Often produces good results with vibrant colors
    """
    # Reshape to pixel array
    data = image.reshape((-1, 3))
    
    def median_cut(pixels, depth=0):
        if depth >= int(np.log2(k)) or len(pixels) == 0:
            # Return average color
            return [np.mean(pixels, axis=0, dtype=np.uint8)]
        
        # Find color channel with greatest range
        ranges = np.ptp(pixels, axis=0)
        channel = np.argmax(ranges)
        
        # Sort by that channel and split at median
        pixels = pixels[pixels[:, channel].argsort()]
        median_idx = len(pixels) // 2
        
        # Recursively process both halves
        return median_cut(pixels[:median_idx], depth + 1) + median_cut(pixels[median_idx:], depth + 1)
    
    # Get palette
    palette = np.array(median_cut(data))
    
    # Assign each pixel to nearest palette color
    distances = np.linalg.norm(data[:, np.newaxis] - palette, axis=2)
    labels = np.argmin(distances, axis=1)
    quantized = palette[labels]
    
    return quantized.reshape(image.shape)
