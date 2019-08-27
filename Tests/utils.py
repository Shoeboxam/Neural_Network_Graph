from Neural_Network.optimizer_private import DPMixin

from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use('fivethirtyeight')


def train_utility(environment, optimizer, graph, queue=None, iterations=None):

    i = 0

    def should_continue():
        if iterations is not None and i > iterations:
            return False
        if issubclass(type(optimizer), DPMixin) and optimizer.epsilon_used >= optimizer.epsilon:
            return False

        return True

    while should_continue():

        i += 1
        sample = environment.sample(quantity=20)

        optimizer.iterate(sample)

        if i % 50 == 0:
            survey = environment.survey()

            if queue:
                queue.put({
                    'iteration': i,
                    'dataset': survey,
                    'produce': {'predicted': graph(survey)}
                })

    survey = environment.survey()
    prediction = graph(survey)
    return environment.error(list(survey.values())[-1], prediction)


def plot_utility(*args):
    process = Process(target=plot_task, args=args)
    process.start()

    return process


def plot_task(plotters, environment, plotting_queue):
    figure = plt.figure()
    figure.subplots_adjust(left=.1, wspace=.3, hspace=.6)
    survey = environment.survey()

    for plotter in plotters:
        plotter.initialize(survey, figure)

    def animate(_):

        plot_step = plotting_queue.get()

        # keep last element from queue
        while not plotting_queue.empty():
            plot_step = plotting_queue.get()

        artists = []
        for plotter in plotters:
            artists.extend(plotter.update(plot_step))

        return artists

    # NOTE: must be saved to a variable, even if not used, for the plot to update
    keep_me = animation.FuncAnimation(
        figure, animate,
        interval=20,
        blit=all(plotter.blit for plotter in plotters))

    print('test')

    plt.show()


def pytest_utility(environment, optimizer, graph, plotters, iterations=None):

    queue = None
    plot_process = None
    if plotters:
        queue = Queue()
        plot_process = plot_utility(plotters, environment, queue)

    try:
        return train_utility(environment, optimizer, graph, queue=queue, iterations=iterations)

    except KeyboardInterrupt:
        if plot_process:
            plot_process.terminate()

        survey = environment.survey()
        prediction = graph(survey)
        return environment.error(list(survey.values())[-1], prediction)

    except Exception as exc:
        if plot_process:
            plot_process.terminate()
        raise exc
