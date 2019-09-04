import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

        self.counter_button = tk.Button(self, text="Count +1", command=self.count)
        self.counter_button.pack()

        self.curr_count_disp = tk.Label(self, text="No counts yet")
        self.curr_count_disp.pack()

    def count(self):
        self.master.count+=1
        self.curr_count_disp['text'] = str(self.master.count)
        print("Added 1 to counter")

root = tk.Tk()
root.count = 0
app = Application(master=root)
app.master.title("Testing")
app.mainloop()