import pyfiglet
import matplotlib.pyplot as plt

from Neural_Network import *


np.set_printoptions(suppress=True, linewidth=10000)
plt.style.use('fivethirtyeight')
plot_points = []

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

        self.stimuli = []
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
            self.stimuli.append(np.array(processed))

        self.character_set = {idx: char for idx, char in enumerate(self.character_set)}

        # Convert stimuli to padded categorical flattened arrays
        for idx in range(len(self.stimuli)):
            for classidx, char_class in self.character_set.items():
                self.stimuli[idx][np.where(self.stimuli[idx] == char_class)] = classidx

            self.stimuli[idx] = np.pad(self.stimuli[idx], ((0, 0), (0, self.max_width - self.stimuli[idx].shape[1])), 'constant')
            self.stimuli[idx] = self.stimuli[idx].flatten()

        self.stimuli = np.array(self.stimuli)
        self.expected = np.eye(len(ascii_vals))

    @Environment._tag
    def sample(self, quantity=1):
        x = np.random.randint(len(self.ascii_vals), size=quantity)

        if self.noise:
            generated_noise = np.random.normal(0., scale=len(self.character_set) // 2, size=self.stimuli[x].shape).astype(int)
            mask = np.random.binomial(1, self.noise, size=self.stimuli[x].shape)
            stimuli = np.mod(self.stimuli[x] + generated_noise * mask, len(self.character_set))
        else:
            stimuli = self.stimuli[x]

        return [stimuli.T, self.stimuli[x].T]

    @Environment._tag
    def survey(self, quantity=None):
        if not quantity:
            quantity = len(self.ascii_vals)
        # x = np.linspace(0, len(self.ascii_vals) - 1, quantity).astype(int)  # Size changes error granularity
        x = np.random.randint(len(self.ascii_vals), size=quantity)

        if self.noise:
            generated_noise = np.random.normal(0., scale=len(self.character_set) // 2, size=self.stimuli[x].shape).astype(int)
            mask = np.random.binomial(1, self.noise, size=self.stimuli[x].shape)
            stimuli = np.mod(self.stimuli[x] + generated_noise * mask, len(self.character_set))
        else:
            stimuli = self.stimuli[x]

        print("Trial:")
        print(self.reformat(stimuli))

        return [stimuli[None].T, self.stimuli[x][None].T]

    def output_nodes(self, tag=None):
        return np.size(self.stimuli[0])

    def plot(self, plt, predict):
        # Do not attempt to plot an image
        pass

    def error(self, expect, predict):
        if self.autoencoder:
            x = np.random.randint(0, expect.shape[1])
            print(self.reformat(predict[:, x]))
            print(self.reformat(expect[:, x]))
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


ascii_vals = [i for i in range(33, 126)]
environment = FigletFonts('banner3', noise=0.0, autoencoder=True, ascii_vals=ascii_vals)


# ~~~ Create the network ~~~
domain = Source(environment, 'stimulus')
codomain = Source(environment, 'expected')

# Layer one
weight_1 = Variable(np.random.normal(size=(100, domain.output_nodes)))
biases_1 = Variable(np.random.normal(size=(100, 1)))
hidden_1 = Bent(weight_1 @ domain + biases_1)

# Layer two
weight_2 = Variable(np.random.normal(size=(environment.output_nodes(), hidden_1.output_nodes)))
biases_2 = Variable(np.random.normal(size=(environment.output_nodes(), 1)))
graph = Logistic(weight_2 @ hidden_1 + biases_2)

# Loss
loss = SumSquared(graph, codomain)

print("Network Summary:")
print(str(graph))
# ~~~ Train the network ~~~
step = .0001
i = 0

while True:
    i += 1
    sample = environment.sample(quantity=10)

    for variable in graph:
        variable -= step * np.average(loss.gradient(sample, variable), axis=2)

    if i % 200 == 0:
        survey = environment.survey(quantity=1)
        prediction = graph(survey)

        error = environment.error(survey['expected'], prediction)
        plot_points.append((i, error))

        # Error plot
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.title('Error')
        plt.plot(*zip(*plot_points), marker='.', color=(.9148, .604, .0945))

        # Environment plot
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.title('Environment')
        environment.plot(plt, prediction)

        plt.pause(0.00001)
