# There is only one environment for the network.
# The environment may produce many sources when sampled.

import abc
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import pandas
import math


class Environment(object):

    # return a random batch of samples from the environment (for training)
    @abc.abstractmethod
    def sample(self, quantity=None):
        pass

    # return a deterministic batch of samples from the environment (for plotting)
    @abc.abstractmethod
    def survey(self, quantity=None):
        pass

    @abc.abstractmethod
    def output_nodes(self, tag):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class Plot(object):
    blit = False

    def __init__(self, position):
        self.position = position
        self.axis = None
        self.components = []

    def initialize(self, survey, figure):
        pass

    def update(self, data):
        return self.components


class PlotError(Plot):
    blit = True

    def __init__(self, position, error):
        super().__init__(position)

        self.error = error

        self.error_x = []
        self.error_y = []
        self.error_y_lim = None

    def initialize(self, survey, figure):

        self.axis = figure.add_subplot(self.position)
        self.axis.set_title('error')

        line_error, = self.axis.plot(
            self.error_x, self.error_y,
            marker='.', color=(.9148, .604, .0945))

        self.components = (line_error,)

    def update(self, data):

        line_error, = self.components
        error_current = self.error(list(data['dataset'].values())[-1], data['produce']['predicted'])
        if math.isnan(error_current):
            return ()

        self.error_x.append(data['iteration'])
        self.error_y.append(error_current)

        line_error.set_data(self.error_x, self.error_y)

        if self.error_y_lim:
            self.error_y_lim[0] = min(self.error_y_lim[0], error_current)
            self.error_y_lim[1] = max(self.error_y_lim[1], error_current)
        else:
            self.error_y_lim = [error_current, error_current + .0001]

        self.axis.set_xlim([0, data['iteration']])
        self.axis.set_ylim(self.error_y_lim)

        return self.components


class PlotLine(Plot):

    def __init__(self, position, layers=None, xlabel=None, ylabel=None, zlabel=None):
        super().__init__(position)
        self.viewpoint = None
        self.layers = layers
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    def initialize(self, survey, figure):

        if not self.layers:
            self.layers = [
                {
                    'x': {'source': 'dataset', 'tag': list(survey.keys())[+0], 'column': 0},
                    'y': {'source': 'dataset', 'tag': list(survey.keys())[-1], 'column': 0},
                    'style': {'color': (0.3559, 0.7196, 0.8637)}
                },
                {
                    'x': {'source': 'dataset', 'tag': list(survey.keys())[+0], 'column': 0},
                    'y': {'source': 'produce', 'tag': 'predicted', 'column': 0},
                    'style': {'color': (.9148, .604, .0945)}
                }
            ]
            if list(survey.values())[-1].shape[1] > 1:
                self.layers[0]['z'] = {'source': 'dataset', 'tag': list(survey.keys())[-1], 'column': 1}
                self.layers[1]['z'] = {'source': 'produce', 'tag': 'predicted', 'column': 1}
            # elif list(survey.values())[0].shape[1] > 1:
            #     pass

        x, *_, y = list(survey.values())

        def get_data(channel):
            if channel['source'] == 'produce':
                return list(survey.values())[0][:, 0, 0] * 0
            return survey[channel['tag']][:, channel['column'], 0]

        # Output of function is 1 dimensional
        if 'z' not in self.layers[0]:

            self.axis = figure.add_subplot(self.position)
            # axis_environment.set_ylim(self._range[0])
            for layer in self.layers:
                self.components.append(self.axis.plot(
                    get_data(layer['x']), get_data(layer['y']),
                    **layer.get('style', {}))[0])

        # Output of function has arbitrary dimensions
        else:
            self.axis = figure.add_subplot(self.position, projection='3d')

            for layer in self.layers:
                self.components.append(self.axis.plot(
                    get_data(layer['x']), get_data(layer['y']), get_data(layer['z']),
                    **layer.get('style', {}))[0])

            self.viewpoint = np.random.randint(0, 360)
            self.axis.view_init(elev=10., azim=self.viewpoint)

        if self.xlabel:
            self.axis.set_xlabel(self.xlabel)

        if self.ylabel:
            self.axis.set_ylabel(self.ylabel)

        if self.zlabel:
            self.axis.set_zlabel(self.zlabel)
        self.axis.set_title('environment')

    def update(self, data):

        def get_data(channel):
            if channel['source'] not in ['dataset', 'produce']:
                raise ValueError("source must be in ['dataset', 'produce']")

            return data[channel['source']][channel['tag']][:, channel['column'], 0]

        for layer, component in zip(self.layers, self.components):
            component.set_data(np.column_stack(
                (get_data(layer['x']), get_data(layer['y']))
            ).T)

        else:
            for layer, component in zip(self.layers, self.components):
                component.set_3d_properties(get_data(layer['z']))
            self.axis.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 1

        return self.components


class PlotScatter(Plot):

    def __init__(self, position, layers=None, xlabel=None, ylabel=None, zlabel=None):
        super().__init__(position)
        self.viewpoint = None
        self.layers = layers
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    def initialize(self, survey, figure):

        if not self.layers:
            self.layers = [
                {
                    'x': {'source': 'dataset', 'tag': list(survey.keys())[+0], 'column': 0},
                    'y': {'source': 'dataset', 'tag': list(survey.keys())[-1], 'column': 0},
                    'style': {'color': (0.3559, 0.7196, 0.8637)}
                },
                {
                    'x': {'source': 'dataset', 'tag': list(survey.keys())[+0], 'column': 0},
                    'y': {'source': 'produce', 'tag': 'predicted', 'column': 0},
                    'style': {'color': (.9148, .604, .0945)}
                }
            ]
            if list(survey.values())[-1].shape[1] > 1:
                self.layers[0]['z'] = {'source': 'dataset', 'tag': list(survey.keys())[-1], 'column': 1}
                self.layers[1]['z'] = {'source': 'produce', 'tag': 'predicted', 'column': 1}
            # elif list(survey.values())[0].shape[1] > 1:
            #     pass

        x, *_, y = list(survey.values())

        def get_data(channel):
            if channel['source'] == 'produce':
                return list(survey.values())[0][:, 0] * 0
            return survey[channel['tag']][:, channel['column']]

        # Output of function is 1 dimensional
        if 'z' not in self.layers[0]:

            self.axis = figure.add_subplot(self.position)
            # axis_environment.set_ylim(self._range[0])
            for layer in self.layers:
                self.components.append(self.axis.scatter(
                    get_data(layer['x']), get_data(layer['y']),
                    **layer.get('style', {})))

        # Output of function has arbitrary dimensions
        else:
            self.axis = figure.add_subplot(self.position, projection='3d')

            for layer in self.layers:
                self.components.append(self.axis.scatter(
                    get_data(layer['x']), get_data(layer['y']), get_data(layer['z']),
                    **layer.get('style', {})))

            self.viewpoint = np.random.randint(0, 360)
            self.axis.view_init(elev=10., azim=self.viewpoint)

        if self.xlabel:
            self.axis.set_xlabel(self.xlabel)

        if self.ylabel:
            self.axis.set_ylabel(self.ylabel)

        if self.zlabel:
            self.axis.set_zlabel(self.zlabel)
        self.axis.set_title('environment')

    def update(self, data):

        def get_data(channel):
            if channel['source'] not in ['dataset', 'produce']:
                raise ValueError("source must be in ['dataset', 'produce']")

            return data[channel['source']][channel['tag']][:, channel['column'], 0]

        if 'z' not in self.layers[0]:
            for layer, component in zip(self.layers, self.components):
                component.set_offsets(np.column_stack(
                    (get_data(layer['x']), get_data(layer['y']))
                ))

        else:
            for layer, component in zip(self.layers, self.components):
                component._offsets3d = (get_data(layer['x']), get_data(layer['y']), get_data(layer['z']))
            self.axis.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 1

        return self.components


class PlotHeatmap(Plot):
    blit = True

    def __init__(self, position, target_labels=None, classes=None):
        super().__init__(position)
        # X labels
        self.targets = target_labels
        # Y labels
        self.classes = classes

    def initialize(self, survey, figure):
        x, *_, y = list(survey.values())
        self.axis = figure.add_subplot(self.position)

        data = []
        for expected in np.swapaxes(y, 1, 0):
            data.append([0] * np.unique(expected, axis=0).shape[0])
        dataframe = pandas.DataFrame([pandas.Series(i) for i in data]).transpose()

        image = self.axis.imshow(
            dataframe,
            cmap='hot', interpolation='nearest',
            animated=True,
            vmin=0, vmax=1)

        x_labels = self.targets or dataframe.columns.values
        y_labels = self.classes or dataframe.index.values

        self.axis.set_xlabel('Target')
        self.axis.set_ylabel('Class')
        self.axis.set_xticks(np.arange(len(x_labels)))
        self.axis.set_yticks(np.arange(len(y_labels)))
        self.axis.set_xticklabels(x_labels)
        self.axis.set_yticklabels(y_labels)

        self.axis.set_title('micro accuracy')

        self.components = (image,)

    def update(self, data):
        dataset = data['dataset']
        prediction = data['produce']['predicted']

        image, = self.components

        x, *_, y = list(dataset.values())

        data = []
        for expected, predicted in zip(np.swapaxes(y, 1, 0), np.swapaxes(prediction, 1, 0)):
            temp = []
            for label in np.unique(expected):
                temp.append(np.mean(label == np.round(predicted)[expected == label], axis=0))
            data.append(temp)
        dataframe = pandas.DataFrame([pandas.Series(i) for i in data]).transpose()
        image.set_array(dataframe)

        return self.components
