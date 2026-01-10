"""Fast image encoding/decoding with msgpack"""
import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image
import base64
m.patch()

def encode_image_msgpack(pil_image):
    """
    Convert PIL Image to msgpack-encoded base64 string
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        str: Base64-encoded msgpack bytes
    """
    img_array = np.array(pil_image)
    packed = msgpack.packb(img_array, use_bin_type=True)

    return base64.b64encode(packed).decode('utf-8')

def decode_image_msgpack(encoded_data):
    """
    Convert msgpack-encoded base64 string back to PIL Image
    
    Args:
        encoded_data: Base64-encoded msgpack bytes
        
    Returns:
        PIL.Image: Decoded image
    """
    packed = base64.b64decode(encoded_data)
    img_array = msgpack.unpackb(packed, raw=False)
    
    return Image.fromarray(img_array)

def encode_image_msgpack_with_compression(pil_image, max_size=1024):
    """
    Encode with optional resizing for faster transmission
    
    Args:
        pil_image: PIL Image object
        max_size: Maximum dimension (width or height)
        
    Returns:
        str: Base64-encoded msgpack bytes
    """
    if max(pil_image.size) > max_size:
        pil_image = pil_image.copy()
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return encode_image_msgpack(pil_image)