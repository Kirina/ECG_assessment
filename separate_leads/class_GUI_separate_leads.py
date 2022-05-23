# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:01:24 2022

@author: kirin
"""
import tkinter as tk
from  tkinter import ttk
import pandas as pd
import wfdb.io
import pathlib
from AlgorithmsV5_k import processing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

class GUI_app(tk.Frame):
    def __init__(self, master, ECG, sampling_frequency, num_leads, table_list, buttons_width, buttons_height, 
                 SNR_threshold, signal_freq_band, window_length, heart_rate_limits, max_loss_passband, min_loss_stopband, length_recording):
        super().__init__(master)
        self.master = master
        self.ECG = ECG
        self.sampling_frequency = sampling_frequency
        self.num_leads = num_leads
        self.table_list = table_list
        self.buttons_width = buttons_width
        self.buttons_height = buttons_height
        
        self.SNR_threshold = SNR_threshold
        self.signal_freq_band = signal_freq_band
        self.window_length = window_length
        self.heart_rate_limits = heart_rate_limits
        self.max_loss_passband = max_loss_passband
        self.min_loss_stopband = min_loss_stopband
        self.length_recording = length_recording

        self.lead_selection = None
        self.import_data_button = None
        self.process_button = None
        self.fig = None
        self.canvas = None

        self.init_plot()
        self.gui_window()  
        
        self.master.bind('<Key-F11>', self.get_fullscreen)
        self.master.bind('<Escape>', self.exit_fullscreen)
           
    def get_fullscreen(self, event):
        self.master.attributes("-fullscreen", True)
        
    def exit_fullscreen(self, event):
        self.master.attributes("-fullscreen", False)
        
    def init_plot(self):
        self.fig = Figure(figsize=(9.6, 3.3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.master)
        self.canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

    def gui_window(self):
        self.plot_figure(self.ECG)
        self.button_create()


    def button_create(self):
        self.import_data_button = tk.Button(self.master, text="Import Data", command=self.import_data, bg = '#E34234')
        self.import_data_button.place(x = 0, y = 0, width = self.buttons_width, height = self.buttons_height)

        self.process_button = tk.Button(self.master, text="Process", command=self.process_ecg, bg = '#0476D0')
        self.process_button.place(x = self.buttons_width, y = 0, width = self.buttons_width, height = self.buttons_height)

        self.lead_selection = tk.Scale(self.master, from_= 1, to_= self.num_leads, command=self.plot_figure, bd = 2, orient="horizontal", bg = '#FFFFF0')
        self.lead_selection.place(x = 0, y = self.buttons_height + 20, width = self.buttons_width * 2)
        self.lead_selection.set(1)
        self.lead_label = tk.Label(self.master, text="Lead: ", font="25", bg = '#FFFFF0')
        self.lead_label.place(x = 0, y = self.buttons_height, width = self.buttons_width * 2)
             
    def import_data(self):
        self.ECG.clear()
    
        file_path = tk.filedialog.askopenfilename()
        file_extension = pathlib.Path(file_path).suffix
        if file_extension == '.txt':
            new = np.loadtxt(file_path, delimiter=",", dtype="int")
            new = np.transpose(new)
        elif file_extension == '.csv':
            new = np.genfromtxt(file_path, delimiter=",", dtype="int")
            new = np.transpose(new)
        elif file_extension == '.hea' or file_extension == '.xws' or file_extension == '.dat' or file_extension == '.atr':
            print(file_path.replace(file_extension, ''))
            new, temp = wfdb.io.rdsamp(file_path.replace(file_extension, ''))
            new = np.transpose(new)
            new = new * 1000
            new = new.astype(int)
        elif file_extension == '.xls':
            new = pd.read_excel(file_path, dtype="int").to_numpy()
            new = np.transpose(new)
        elif file_extension == '.xlsx':
            new = pd.read_excel(file_path, dtype="int").to_numpy()
            new = np.transpose(new)
        else:
            print('No file selected or file extension not accepted')
            print('Please only use files with extension: -.txt -.csv -.hea -.xls -.xlsx')
            return
        for lead in range(0, self.num_leads + 1):
            self.ECG.append(new[lead])
        self.plot_figure(self.ECG)
        
    def plot_figure(self, _):
        if not len(self.ECG) == 0:
            self.master.wm_title("Embedding in Tk")
            self.fig.clear()
            ax = self.fig.add_subplot()

            ax.plot(self.ECG[self.lead_selection.get()])
            ax.set_xlabel("time [s]")
            ax.set_ylabel("f(t)")
            ax.minorticks_on()
            
            original_stepsize = len(self.ECG[0]) / 5
            length_ecg_data = len(self.ECG[0])
            original_x_axis = [0, original_stepsize, length_ecg_data - 3 * original_stepsize, length_ecg_data - 2 * original_stepsize, length_ecg_data - original_stepsize, length_ecg_data]
            step_size = int(self.length_recording / (len(original_x_axis) - 1))
            ax.set_xticks(original_x_axis, labels=list(range(0, self.length_recording + step_size, step_size)))
            
            ax.grid(which='major', linestyle='-', linewidth='0.4', color='red')
            ax.grid(which='minor', linestyle='-', linewidth='0.4', color=(1, 0.7, 0.7))
    
            self.canvas.draw()
            
            self.process_button.lift()
            self.import_data_button.lift()
            self.lead_selection.lift()
            self.lead_label.lift()
            
    def process_ecg(self):
        # global sampling_freq
        if not self.ECG:
            return
        else:
            res = (processing(self.ECG, self.num_leads, self.sampling_frequency, self.SNR_threshold, self.signal_freq_band, self.window_length, 
                           self.heart_rate_limits, self.max_loss_passband, self.min_loss_stopband, self.sampling_frequency, self.length_recording))
            for x in range(1, len(self.table_list)):
                for y in range(0, self.num_leads):
                    self.table_list[x][y + 1] = res[x - 1][y]
                for y in range(self.num_leads, len(self.table_list[0]) - 1):
                    self.table_list[x][y + 1] = ""
            self.table_fill()

    def table_fill(self):
        table_labels = ttk.Treeview(self.master, height=len(self.table_list) - 1)
        table_labels['columns'] = self.table_list[0][0]
        table_labels.column('#0', width = 0, stretch = tk.NO)
        table_labels.column(self.table_list[0][0], anchor = tk.CENTER, width =  int(self.buttons_width * 1.5))
        table_labels.heading(self.table_list[0][0], text = self.table_list[0][0], anchor = tk.CENTER)
        
        
        for i in range(1, len(self.table_list)):
            table_labels.insert(parent = '',index = 'end', iid = i, text = '', values = (self.table_list[i][:1]))
        table_labels.place(x = self.buttons_width * 2, y = 0)
        
        table = ttk.Treeview(self.master, height=len(self.table_list) - 1)
        table['columns'] = self.table_list[0][1:]
        table.column('#0', width = 0, stretch = tk.NO)
        for col in self.table_list[0]: 
            # Column layout
            if col == self.table_list[0][0]:
                pass
            else: 
                table.column(col, anchor = tk.CENTER, width = int(self.buttons_width / 2.5))
                table.heading(col, text = col, anchor = tk.CENTER)
        for i in range(1, len(self.table_list)):
            table.insert(parent = '',index = 'end', iid = i, text = '', values = (self.table_list[i][1:]))
        table.place(x = self.buttons_width * 2 + int(self.buttons_width * 1.5), y = 0)

def get_curr_screen_geometry():
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    width = int(geometry.split('x')[0])
    height = int(geometry.split('x')[1].split('+')[0])
    root.destroy()
    return width - 50, height - 50


# # region set parameters
sampling_frequency = 500        # Hz
max_loss_passband = 0.1     # dB
min_loss_stopband = 20      # dB
SNR_threshold = 0.5
signal_freq_band = [2, 40]      # from .. to .. in Hz
heart_rate_limits = [24, 300]       # from ... to ... in beats per minute
length_recording = 10       # seconds
window_length = 100   

screen_width, screen_height = get_curr_screen_geometry()
init_gui_width = int(screen_width / 1.5)
init_gui_height = int(screen_height / 1.5)
buttons_width = int(screen_width / 12.5)
buttons_height = int(screen_height / 35)
num_leads = 12
ECG = []
table_list = [["Lead", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
       ["Stationary Signal Check", "", "", "", "", "", "", "", "", "", "", "", ""],
       ["Heart Rate Check", "", "", "", "", "", "", "", "", "", "", "", ""],
       ["SNR Check", "", "", "", "", "", "", "", "", "", "", "", ""],
       ["Overall Result", "", "", "", "", "", "", "", "", "", "", "", ""]]

print(f'{screen_width} {screen_height}')
print(f'{init_gui_width} {init_gui_height}')

root = tk.Tk()
root.geometry(f"{init_gui_width}x{init_gui_height}")

myapp = GUI_app(root, ECG, sampling_frequency, num_leads, table_list, buttons_width, buttons_height, SNR_threshold, 
                signal_freq_band, window_length, heart_rate_limits, max_loss_passband, min_loss_stopband, length_recording)
myapp.mainloop()
