
from Neural_Network import *
from Environments.Image import ImageSingle

from PIL import Image


sharpen = np.array(
    [[+0, -1, 0],
     [-1, +5, -1],
     [+0, -1, 0]])

gaussian = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]]) / 16

identity = np.array(
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]])

edge = np.array(
    [[+0, 0, 0],
     [-1, 0, 1],
     [+0, 0, 0]])

emboss = np.array(
    [[-2, -1, 0],
     [-1, 1, 1],
     [+0, 1, 2]])


def test_image():
    # image_path = 'images/Bikesgray.jpg'
    image_path = 'images/sample_wheat.png'

    # Define environment
    image = ImageSingle(image_path)

    # Create entry point for image in network
    source = Source(image, 'stimulus')

    # Add a convolution layer
    convolution = Convolve(source, gaussian)

    # Call network with a sample image
    convolved = convolution(image.sample())

    # Convert output back into image
    Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8)).show()
