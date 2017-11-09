from Neural_Network import *

from PIL import Image

# path = 'images/Bikesgray.jpg'
image_path = 'images/sample_wheat.png'

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


class ImageSingle(Environment):
    def __init__(self, path):
        self.data = np.array(Image.open(path)).astype(float)

    @Environment._tag
    def sample(self, quantity):
        return [self.data, np.array(1)]

    @Environment._tag
    def survey(self, quantity):
        return [self.data, np.array(1)]

    def output_nodes(self, tag):
        if tag is 'stimulus':
            return self.data.shape
        return [1]

    def plot(self, plt, predict):
        plt.imshow(predict)


# Create convolution gate
image = ImageSingle(image_path)
source = Source(image, image.tags[0])
convolution = Convolve(source, gaussian)
convolved = convolution(image.sample(tagged=True))


Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8)).show()
