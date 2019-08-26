# There is only one environment for the network.
# The environment may produce many sources when sampled.

import abc
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3


class Environment(object):
    blit = False

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
    def plot_initialize(self, survey, figure):
        pass

    @abc.abstractmethod
    def plot_update(self, survey, prediction, axis, components):
        pass


class PlotMixin(object):
    blit = True

    def plot_initialize(self, survey, figure):

        x, *_, y = list(survey.values())

        # Output of function is 1 dimensional
        if y.shape[1] == 1:

            axis_environment = figure.add_subplot(1, 2, 2)
            # axis_environment.set_ylim(self._range[0])
            scatter_actuals = axis_environment.plot(x[:, 0], y[:, 0], marker='.', color=(0.3559, 0.7196, 0.8637))
            scatter_predict = axis_environment.plot(x[:, 0], y[:, 0] * 0, marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        elif y.shape[1] > 1:
            axis_environment = figure.add_subplot(1, 2, 2, projection='3d')

            scatter_actuals = axis_environment.plot(x[:, 0, 0], y[:, 0, 0], y[:, 1, 0], color=(0.3559, 0.7196, 0.8637))
            scatter_predict = axis_environment.plot(x[:, 0, 0], y[:, 0, 0] * 0, y[:, 1, 0] * 0, color=(.9148, .604, .0945))

            self.viewpoint = np.random.randint(0, 360)
            axis_environment.view_init(elev=10., azim=self.viewpoint)

        else:
            raise ValueError('y must be non-empty')

        axis_environment.set_title('Environment')

        return {
            'axis': axis_environment,
            'components': (scatter_actuals[0], scatter_predict[0])
        }

    def plot_update(self, survey, prediction, axis, components):
        scatter_actuals, scatter_predict = components
        x, *_, y = list(survey.values())

        scatter_actuals.set_data(np.column_stack((x[:, 0, 0], y[:, 0, 0])).T)
        scatter_predict.set_data(np.column_stack((x[:, 0, 0], prediction[:, 0, 0])).T)

        # Output of function has arbitrary dimensions
        if y.shape[1] > 1:
            scatter_actuals.set_3d_properties(y[:, 1, 0])
            scatter_predict.set_3d_properties(prediction[:, 1, 0])
            axis.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5


class ScatterMixin(object):

    def plot_initialize(self, survey, figure):

        x, *_, y = list(survey.values())

        # Output of function is 1 dimensional
        if y.shape[1] == 1:

            axis_environment = figure.add_subplot(1, 2, 2)
            # axis_environment.set_ylim(self._range[0])
            scatter_actuals = axis_environment.scatter(x[:, 0], y[:, 0], marker='.', color=(0.3559, 0.7196, 0.8637))
            scatter_predict = axis_environment.scatter(x[:, 0], y[:, 0] * 0, marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        elif y.shape[1] > 1:
            axis_environment = figure.add_subplot(1, 2, 2, projection='3d')

            scatter_actuals = axis_environment.scatter(x[:, 0], y[:, 0], y[:, 1], color=(0.3559, 0.7196, 0.8637))
            scatter_predict = axis_environment.scatter(x[:, 0], y[:, 0] * 0, y[:, 1] * 0, color=(.9148, .604, .0945))

            self.viewpoint = np.random.randint(0, 360)
            axis_environment.view_init(elev=10., azim=self.viewpoint)

        else:
            raise ValueError('y must be non-empty')

        axis_environment.set_title('Environment')

        return {
            'axis': axis_environment,
            'components': (scatter_actuals, scatter_predict)
        }

    def plot_update(self, survey, prediction, axis, components):
        scatter_actuals, scatter_predict = components
        x, *_, y = list(survey.values())

        # Output of function is 1 dimensional
        if y.shape[1] == 1:
            scatter_actuals.set_offsets(np.column_stack((x[:, 0], y[:, 0])))
            scatter_predict.set_offsets(np.column_stack((x[:, 0], prediction[:, 0])))

        # Output of function has arbitrary dimensions
        elif y.shape[1] > 1:

            scatter_actuals._offsets3d = (x[:, 0, 0], y[:, 0, 0], y[:, 1, 0])
            scatter_predict._offsets3d = (x[:, 0, 0], prediction[:, 0, 0], prediction[:, 1, 0])
            axis.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5


class HeatmapMixin(object):
    def init_plot(self, figure):
        return ()

    def plot(self, plt, predict, components):
        pass
