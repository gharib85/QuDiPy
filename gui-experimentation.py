import tkinter as tk
import numpy as np
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class Application(tk.Frame):
    def __init__(self, master=None, num=10):
        super().__init__(master)
        self.master = master
        self.continue_plotting = True
        self.num = num

        self.create_widgets()
        self.pack()

    def create_widgets(self):

        # Creating Figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")

        self.graph = FigureCanvasTkAgg(self.fig, master=root)
        self.graph.get_tk_widget().pack(side="left")

        self.stop_live_plotting_button = tk.Button(self, text="QUIT", fg='red', command=self.stop_live_plotting)
        self.stop_live_plotting_button.pack(side="bottom")

        self.increase_number_of_points_button = tk.Button(self, text="Increase number of points",
                                                            command=self.increase_number_of_points)
        self.increase_number_of_points_button.pack()

        self.decrease_number_of_points_button = tk.Button(self, text="Decrease number of points",
                                                            command=self.decrease_number_of_points)
        self.decrease_number_of_points_button.pack()

        self.display_number_of_points_label = tk.Label(self, text=str(self.num))
        self.display_number_of_points_label.pack()

    def plot(self, data_x, data_y):
            self.ax.cla()
            self.ax.grid()
            self.ax.scatter(data_x, data_y)
            self.graph.draw()

    def stop_live_plotting(self):
        self.continue_plotting = False

    def increase_number_of_points(self):
        self.num+=1
        self.update_number_of_points()

    def decrease_number_of_points(self):
        self.num-=1
        self.update_number_of_points()

    def update_number_of_points(self):
        self.display_number_of_points_label['text'] = str(self.num)

        

root = tk.Tk()
root.count = 0
app = Application(master=root)
app.master.title("Testing")

while app.continue_plotting is True:
    x = np.random.rand(app.num)
    y = np.random.rand(app.num)
    app.plot(x,y)
    app.update()
    app.update_idletasks()