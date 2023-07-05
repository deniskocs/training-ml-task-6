import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        self.fig = plt.figure(figsize=(24, 8))

        self.ax = self.fig.add_subplot()

    def drawErrors(self, errors):
        self.ax.cla()
        self.ax.plot(range(len(errors)), errors, 'r')

    def flush(self):
        plt.draw()
        plt.pause(0.02)
