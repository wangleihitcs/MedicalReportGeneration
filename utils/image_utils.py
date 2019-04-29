from PIL import Image
import numpy as np

# get image pixels value
def getImages(image_path, image_size):
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image = image_rgb.resize((image_size, image_size), Image.ANTIALIAS)
    image_array = np.array(image) / 255.0
    return image_array