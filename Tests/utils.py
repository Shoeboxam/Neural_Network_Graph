import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('fivethirtyeight')

import numpy as np


def train_utility(environment, loss, graph, plot=False, iterations=None):
    # ~~~ Train the network ~~~

    plot_points = []
    step = .01
    i = 0

    while True:
        if iterations is not None and i > iterations:
            break

        i += 1
        sample = environment.sample(quantity=20)

        for variable in graph:
            print('VARIABLE STEP:')
            print(variable.label)
            grad = loss.gradient(sample, variable)

            print('final shapes:')
            print(grad.shape)
            print(variable.shape)
            print(i)
            variable -= step * np.average(grad, axis=0)

        if i % 50 == 0:
            survey = environment.survey()
            print('forward:')
            print({identifier: survey[identifier].shape for identifier in survey})
            prediction = graph(survey)

            error = environment.error(survey['expected'], prediction)
            plot_points.append((i, error))

            if plot:
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

    survey = environment.survey()
    prediction = graph(survey)
    return environment.error(survey['expected'], prediction)
