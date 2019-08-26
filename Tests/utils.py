import numpy as np
from multiprocessing import Process

import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('fivethirtyeight')


def train_utility(environment, loss, graph, queue=None, iterations=None, step=.01):
    # ~~~ Train the network ~~~

    i = 0

    while True:
        if iterations is not None and i > iterations:
            break

        i += 1
        sample = environment.sample(quantity=20)

        for variable in graph:
            # print('VARIABLE STEP:')
            # print(variable.label)
            grad = loss.gradient(sample, variable)

            # print('final shapes:')
            # print(grad.shape)
            # print(variable.shape)
            # print(i)
            variable -= step * np.average(grad, axis=0)

        if i % 50 == 0:
            survey = environment.survey()
            prediction = graph(survey)
            error = environment.error(list(survey.values())[-1], prediction)

            if queue:
                queue.put({
                    'error': (i, error),
                    'environment': {
                        'survey': survey, 'prediction': prediction
                    }
                })

    survey = environment.survey()
    prediction = graph(survey)
    return environment.error(list(survey.values())[-1], prediction)


def plot_utility(*args):
    process = Process(target=plot_task, args=args)
    process.start()

    return process


def plot_task(environment, plotting_queue):
    figure = plt.figure()

    axis_error = figure.add_subplot(1, 2, 1)
    axis_error.set_title('Error')

    error_x, error_y = [], []
    error_y_lim = None
    line_error, = axis_error.plot(error_x, error_y, marker='.', color=(.9148, .604, .0945))

    environment_plot = environment.plot_initialize(environment.survey(), figure)

    def animate(_):
        nonlocal error_y_lim

        plot_step = plotting_queue.get()

        # keep last element from queue
        while not plotting_queue.empty():
            plot_step = plotting_queue.get()

        if 'error' in plot_step:
            error_x.append(plot_step['error'][0])
            error_y.append(plot_step['error'][1])
            line_error.set_data(error_x, error_y)

            if error_y_lim:
                error_y_lim[0] = min(error_y_lim[0], plot_step['error'][1])
                error_y_lim[1] = max(error_y_lim[1], plot_step['error'][1])
            else:
                error_y_lim = [plot_step['error'][1], plot_step['error'][1]]

            axis_error.set_xlim([0, plot_step['error'][0]])
            axis_error.set_ylim(error_y_lim)

        if 'environment' in plot_step:
            environment.plot_update(**{**plot_step['environment'], **environment_plot})

        return (line_error, *environment_plot['components'])

    # NOTE: must be saved to a variable, even if not used, for the plot to update
    keep_me = animation.FuncAnimation(
        figure, animate,
        interval=20,
        blit=environment.blit)

    plt.show()
