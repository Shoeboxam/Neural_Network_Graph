import pandas
from Neural_Network import *


class Dataset(Environment):

    def __init__(self, **dataframes):
        lengths = set(len(dataframes[tag]) for tag in dataframes)
        if len(lengths) != 1:
            raise ValueError('dataframes do not have equal size')

        self._number_rows = next(iter(lengths))
        self._dataframes = dataframes

        self.viewpoint = np.random.randint(0, 360)

    def sample(self, quantity=None):
        quantity = quantity or 1
        indices = np.random.randint(0, self._number_rows, size=quantity)
        return {tag: self._dataframes[tag][indices] for tag in self._dataframes}

    def survey(self, quantity=None):
        quantity = min(quantity or 128, self._number_rows)
        indices = np.floor(np.linspace(0, self._number_rows, quantity))
        return {tag: self._dataframes[tag][indices] for tag in self._dataframes}

    def output_nodes(self, tag):
        return self._dataframes[tag].shape[1]

    def plot(self, plt, predict):
        survey = self.survey()
        x = survey[0]
        y = survey[1]

        # Output of function is 1 dimensional
        if y.shape[0] == 1:
            ax = plt.subplot(1, 2, 2)
            plt.ylim(self._range[0])

            ax.plot(x[0], y[0], marker='.', color=(0.3559, 0.7196, 0.8637))
            ax.plot(x[0], predict[0], marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        if y.shape[0] > 1:

            ax = plt.subplot(1, 2, 2, projection='3d')
            plt.title('Environment')
            ax.scatter(x[0], y[0], y[1], color=(0.3559, 0.7196, 0.8637))
            ax.scatter(x[0], predict[0], predict[1], color=(.9148, .604, .0945))
            ax.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)


dataframe = pandas.read_csv('/home/shoe/Desktop/MaPUMS5full.csv')

environment = Dataset(
    stimulus=dataframe[['educ']].to_numpy(),
    expected=dataframe[['married']].to_numpy())

domain = Source(environment, 'stimulus')
codomain = Source(environment, 'expected')

# Layer one
weight_1 = Variable(np.random.uniform(size=(4, domain.output_nodes)))
biases_1 = Variable(np.random.uniform(size=(4, 1)))
hidden_1 = Bent(weight_1 @ domain + biases_1)

# Layer two
weight_2 = Variable(np.random.uniform(size=(2, hidden_1.output_nodes)))
biases_2 = Variable(np.random.uniform(size=(2, 1)))
graph = Bent(weight_2 @ hidden_1 + biases_2)

# Loss
loss = SumSquared(graph, codomain)
variables = graph.variables

print(loss)

