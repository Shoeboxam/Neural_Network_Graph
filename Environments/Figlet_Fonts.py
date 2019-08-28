import pyfiglet

import numpy as np
from Environments.base import Environment


# Autoencoder for font denoising!

"""     #######
        #       #  ####  #      ###### #####
        #       # #    # #      #        #
        #####   # #      #      #####    #
        #       # #  ### #      #        #
        #       # #    # #      #        #
        #       #  ####  ###### ######   #      """

# http://www.figlet.org/


class FigletFonts(Environment):

    def __init__(self, font='banner', noise=0., autoencoder=False, ascii_vals=None):
        self.noise = noise
        self.autoencoder = autoencoder

        self.data = []
        self.character_set = set()

        self.max_width = 0

        if not ascii_vals:
            ascii_vals = [i for i in range(32, 127)]
        self.ascii_vals = ascii_vals

        # Initial pass to determine letter width
        for letter in ascii_vals:
            raw = pyfiglet.figlet_format(chr(letter), font=font)

            width = len(raw.splitlines()[0])
            self.max_width = self.max_width if self.max_width > width else width

            numerical = []
            for line in raw.splitlines():
                numerical.append([ord(cha) for cha in line])

            processed = np.array(numerical)
            self.character_set.update(np.unique(processed))
            self.data.append(np.array(processed))

        self.character_set = {idx: char for idx, char in enumerate(self.character_set)}

        # Convert stimuli to padded categorical flattened arrays
        for idx in range(len(self.data)):
            for classidx, char_class in self.character_set.items():
                self.data[idx][np.where(self.data[idx] == char_class)] = classidx

            self.data[idx] = np.pad(self.data[idx], ((0, 0), (0, self.max_width - self.data[idx].shape[1])), 'constant')
            self.data[idx] = self.data[idx].flatten()

        self.data = np.array(self.data)
        self.expected = np.eye(len(ascii_vals))

    def sample(self, quantity=1):
        samples = np.random.randint(len(self.ascii_vals), size=quantity)

        if self.noise:
            generated_noise = np.random.normal(
                0., scale=len(self.character_set) // 2,
                size=self.data[samples].shape).astype(int)

            mask = np.random.binomial(
                1, self.noise,
                size=self.data[samples].shape)

            stimulus = np.mod(
                self.data[samples] + generated_noise * mask,
                len(self.character_set))
        else:
            stimulus = self.data[samples]

        stimulus = np.atleast_3d(stimulus)
        expected = np.atleast_3d(self.data[samples])

        return {'stimulus': stimulus, 'expected': expected}

    def survey(self, quantity=None):
        if not quantity:
            quantity = len(self.ascii_vals)
        # samples = np.linspace(0, len(self.ascii_vals) - 1, quantity).astype(int)  # Size changes error granularity
        samples = np.random.randint(len(self.ascii_vals), size=quantity)

        if self.noise:
            generated_noise = np.random.normal(
                0., scale=len(self.character_set) // 2,
                size=self.data[samples].shape).astype(int)

            mask = np.random.binomial(
                1, self.noise,
                size=self.data[samples].shape)

            stimulus = np.mod(
                self.data[samples] + generated_noise * mask,
                len(self.character_set))
        else:
            stimulus = self.data[samples]

        stimulus = np.atleast_3d(stimulus)
        expected = np.atleast_3d(self.data[samples])

        return {'stimulus': stimulus, 'expected': expected}

    def output_nodes(self, tag=None):
        return self.data.shape[-1]

    def plot(self, plt, predict):
        # Do not attempt to plot an image
        pass

    def error(self, expect, predict):
        if self.autoencoder:
            x = np.random.randint(0, expect.shape[1])
            print(self.reformat(predict[x]))
            print(self.reformat(expect[x]))
            return np.linalg.norm(expect - predict)

        predict_id = np.argmax(predict, axis=0)
        expect_id = np.argmax(expect, axis=0)

        return int((1.0 - np.mean((predict_id == expect_id).astype(float))) * 100)

    def reformat(self, data):
        data = np.round(np.clip(data.reshape((-1, self.max_width)), 0, len(self.character_set) - 1)).astype(int)
        ascii_valued = np.zeros(data.shape)
        for classidx, char_class in self.character_set.items():
            ascii_valued[np.where(data == classidx)] = char_class

        output = ''
        for line in ascii_valued:
            output += ''.join([chr(int(round(symbol))) for symbol in line]) + '\n'
        return output
