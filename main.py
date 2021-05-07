
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import csv
import sqlite3 as db
from sqlite3 import Error
import csv_to_sqlite
import numpy as np
import scipy as sp
import scipy.special
import pandas as pd
import pandastable as pds
import itertools
from itertools import zip_longest
import pulp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import time
import bs4 as bs
import urllib3
import requests
import re
from selenium import webdriver
from sqlalchemy import create_engine
from modules import DataFetcher, Progress, NeuralNet, Player, WebScraper, PlayerDisplay, Lineup


########################################################################################################################
class MainDisplay:

    def __init__(self, master):

        # main frame initialization
        self.master = master
        self.master.title('Fan Duel')

        # gets dimensions in order to center app
        width = self.master.winfo_screenwidth()
        height = self.master.winfo_screenheight()

        # vars to center app
        window_width = 1100
        window_height = 800
        main_width = int(width / 2 - window_width / 2)
        main_height = int(height / 2 - window_height / 2)

        # dimensions of app for centering
        self.master.geometry(f'{window_width}x{window_height}+{main_width}+{main_height}')

        self.frame = tk.Frame(self.master)
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)
        self.tree = None

        # input variables
        self.enter = None
        self.text_input = tk.StringVar()
        self.test_year = tk.StringVar()
        self.week = tk.StringVar()
        self.stack1_ckbtn = tk.IntVar()
        self.stack2_ckbtn = tk.IntVar()
        self.stack3_ckbtn = tk.IntVar()
        self.contest_var = tk.StringVar()

        # lineup variables
        self.lineup_tree = None
        self.lineup_object = None

        # graph variables
        self.lineup_ax = None
        self.lineup_figure = None
        self.lineup_tab = None
        self.lineup_graph = None
        self.x_names = []
        self.y_pts = []

        # other variables
        self.datafetcher = DataFetcher()

        # creates games table if it's not already present
        # in order for rest of program to compile
        self.datafetcher.create_games_table()

        # tab initialization
        self.main_tab = ttk.Notebook(self.master)

        # load widgets
        self.create_menubar()
        self.create_table_section()
        self.create_lineup_tab()
        self.create_nn_tab()
        self.create_lineup_object()
        self.graph_lineup()

    def create_menubar(self):

        # declare menubar
        file_menu = tk.Menu(self.menubar)
        selection_menu = tk.Menu(self.menubar)
        algorithms_menu = tk.Menu(self.menubar)
        analysis_menu = tk.Menu(self.menubar)
        quit_menu = tk.Menu(self.menubar)

        # add menu item selections
        self.menubar.add_cascade(label='File', menu=file_menu)
        self.menubar.add_cascade(label='Lineups', menu=selection_menu)
        self.menubar.add_cascade(label='Algorithms', menu=algorithms_menu)
        self.menubar.add_cascade(label='Neural Network', menu=analysis_menu)
        self.menubar.add_cascade(label='Quit', menu=quit_menu)

        # add commands to menu items
        file_menu.add_command(label='Soft Start (csv file)', command=lambda: self.create_new_by_file())
        file_menu.add_command(label='Hard Start (web scrape)', command=lambda: self.create_new_by_scraping())
        quit_menu.add_command(label='Exit', command=self.master.destroy)
        algorithms_menu.add_command(label='Adjust Parameters', command=lambda: self.open_nn_parameters())
        algorithms_menu.add_command(label='Search', command=lambda: self.tune_search_algo())

    def create_lineup_object(self):

        season = self.test_year.get()
        week = self.week.get()

        self.lineup_object = Lineup(season, week)

    def create_table_section(self):

        wrapper1 = tk.LabelFrame(self.master, text="Current Players")
        wrapper1.pack(fill="both", expand="no", padx=5, pady=5)

        # configure tree view (table)
        style = ttk.Style()
        style.theme_use("default")
        self.tree = ttk.Treeview(wrapper1, columns=(1, 2, 3, 4, 5, 6, 7), show="headings", height=6, selectmode="browse")
        self.tree.pack(side=tk.LEFT, fill="both", expand="yes")
        self.tree.heading(1, text="Player")
        self.tree.heading(2, text="Team")
        self.tree.heading(3, text="Position")
        self.tree.heading(4, text="Season")
        self.tree.heading(5, text="Week")
        self.tree.heading(6, text="Salary")
        self.tree.heading(7, text="NN Score")

        # tree scrollbar
        v_scrollbar = tk.Scrollbar(self.tree, orient=tk.VERTICAL, command=self.tree.yview())
        v_scrollbar.pack(side=tk.RIGHT, fill="y", expand=False)
        self.tree.configure(yscrollcommand=v_scrollbar.set)

        # bind double click to open player data dialog
        self.tree.bind("<Double-1>", self.open_player_data)

        # first column is left justified, all others centered
        self.tree.column(column=0, width=100)
        for i in range(2, 8):

            self.tree.column(column=i, anchor='center', width=80)

        # create search canvas to the right of the table so we can use grid to manage buttons
        search_canvas = tk.Canvas(wrapper1)
        search_canvas.pack(side=tk.RIGHT, fill='both', expand=False)

        # initialize spinbox and declare the text var
        self.week.set('1')

        season_options = ["2015",
                          "2015", # listed twice otherwise first value disappears
                          "2016",
                          "2017",
                          "2018",
                          "2019",
                          "2020",
                          "2021"]

        # declare entry, dropdowns, spin boxes, labels, and buttons
        year_drop_down = ttk.OptionMenu(search_canvas, self.test_year, *season_options)
        week_spinbox = ttk.Spinbox(search_canvas, from_=1, to=17, textvariable=self.week)
        label = tk.Label(search_canvas, text="Search")
        self.enter = tk.Entry(search_canvas, textvariable=self.text_input)
        search_btn = tk.Button(search_canvas, text="Search Player", command=lambda: self.search_player())
        clear_btn = tk.Button(search_canvas, text="Clear Search", command=lambda: self.clear_search())
        scheduled_btn = tk.Button(search_canvas, text="Schedule", command=lambda: self.schedule_to_play())
        unschedule_btn = tk.Button(search_canvas, text="Unschedule", command=lambda: self.unschedule_from_play())
        add_to_lineup_btn = tk.Button(search_canvas, text="Add to Lineup", command=lambda: self.add_to_lineup())
        update_btn = tk.Button(search_canvas, text="Update", command=lambda: self.update_player_data())

        # display the widgets
        year_drop_down.grid(row=0, column=0, padx=5, pady=1, sticky='ew')
        week_spinbox.grid(row=1, column=0, padx=5, pady=1, sticky='ew')
        label.grid(row=2, column=0, sticky='ew')
        self.enter.grid(row=3, column=0, padx=5, pady=10, sticky='ew')
        search_btn.grid(row=4, column=0, padx=5, pady=1, stick='ew')
        clear_btn.grid(row=5, column=0, padx=5, pady=1, sticky='ew')
        scheduled_btn.grid(row=6, column=0, padx=5, pady=1, sticky='ew')
        unschedule_btn.grid(row=7, column=0, padx=5, pady=1, sticky='ew')
        add_to_lineup_btn.grid(row=8, column=0, padx=5, pady=1, sticky='ew')
        update_btn.grid(row=9, column=0, padx=5, pady=1, sticky='ew')

        # load default table view
        self.search_player()

    def create_lineup_tab(self):

        # define tabs
        self.main_tab.pack(fill='both', expand='yes')
        self.lineup_tab = ttk.Frame(self.main_tab)
        self.main_tab.add(self.lineup_tab, text='Lineup')
        self.lineup_tree = ttk.Treeview(self.lineup_tab, columns=1, show='headings', selectmode='browse')
        self.lineup_tree.heading(1, text='Lineup')
        self.lineup_tree.column('#1', width=200, stretch=0)
        lineup_canvas = tk.Canvas(self.lineup_tab, bd=1, relief=tk.SUNKEN)

        # show
        self.main_tab.pack(expand=1, fill='both')
        self.lineup_tree.pack(side=tk.LEFT, fill='both', padx=5)
        lineup_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=5)

        # declare labels frames for lineup
        functions_lf = tk.LabelFrame(lineup_canvas, text='Functions')
        stack_lf = tk.LabelFrame(lineup_canvas, text='Stacking')
        contest_lf = tk.LabelFrame(lineup_canvas, text='Contest')

        # show label frames
        functions_lf.pack(fill='both', expand=1, padx=5, pady=5)
        stack_lf.pack(fill='both', expand=1, padx=5, pady=5)
        contest_lf.pack(fill='both', expand=1, padx=5, pady=5)

        # declare function buttons
        gen_lineup_btn = tk.Button(functions_lf, text="Generate Lineup", command=lambda: self.generate_lineup())
        remove_btn = tk.Button(functions_lf, text='Remove from Lineup', command=lambda: self.remove_from_lineup())
        reset_btn = tk.Button(functions_lf, text='Reset Lineup', command=lambda: self.reset_lineup())
        graph_btn = tk.Button(functions_lf, text='Graph', command=lambda: self.update_graph())
        test_btn = tk.Button(functions_lf, text='Test', command=lambda: self.create_new_by_scraping())

        # show function buttons
        gen_lineup_btn.pack(side='top', fill='both', padx=10, pady=1)
        remove_btn.pack(side='top', fill='both', padx=10, pady=1)
        reset_btn.pack(side='top', fill='both', padx=10, pady=1)
        graph_btn.pack(side='top', fill='both', padx=10, pady=1)
        test_btn.pack(side='top', fill='both', padx=10, pady=1)

        # initialize check buttons
        self.stack1_ckbtn = tk.Checkbutton(stack_lf, text='QB/WR Stack')
        self.stack2_ckbtn = tk.Checkbutton(stack_lf, text='RB/DEF Stack')
        self.stack3_ckbtn = tk.Checkbutton(stack_lf, text='QB/TE Stack')

        # show check buttons
        self.stack1_ckbtn.grid(row=1, column=1)
        self.stack2_ckbtn.grid(row=2, column=1)
        self.stack3_ckbtn.grid(row=1, column=2)

        # initialize and show 'contest' section of lineup
        options = ["All", "50/50 (Low Variability)", "Tournament (High Variability)"]
        self.contest_var.set(options[0])
        contest_dropdown = tk.OptionMenu(contest_lf, self.contest_var, *options)
        contest_dropdown.pack(side='top', fill='x', padx=10, pady=1)

    def test_progress(self):

        pass



    def create_nn_tab(self):

        nn_tab = ttk.Frame(self.main_tab)
        self.main_tab.add(nn_tab, text='Neural Network')
        self.main_tab.pack(expand=1, fill='both')

        nn_figure = plt.Figure(figsize=(6, 5), dpi=100)
        ax2 = nn_figure.add_subplot(111)
        graph = FigureCanvasTkAgg(nn_figure, nn_tab)
        graph.get_tk_widget().pack(side='right', fill='both', pady=5, expand=1)

        nn_canvas = tk.Canvas(nn_tab)
        nn_canvas.pack(side='right', fill='both', expand=1,  padx=5, pady=2)

        nn_lf = tk.LabelFrame(nn_canvas, text='Functions')
        nn_lf.pack(fill='both', expand=1, padx=5, pady=5)

        param_lf = tk.LabelFrame(nn_canvas, text='Parameters')
        param_lf.pack(fill='both', expand=1, padx=5, pady=5)

        train_btn = tk.Button(nn_lf, text="Train Network", command=lambda: self.train_players())
        train_btn.grid(row=0, column=0, padx=5, pady=1, sticky='ew')

        learning_rate = tk.IntVar()
        lr_label = tk.Label(param_lf, text='Learning Rate')
        lr_label.pack()

        learning_rate_slider = tk.Scale(param_lf, variable=learning_rate, from_=.1, to=1, resolution=.1,
                                        orient=tk.HORIZONTAL)
        learning_rate_slider.pack(fill='x', padx=10)

    def create_profits_tab(self):
        pass

    def graph_lineup(self):

        # create lineup projected performance graph
        self.lineup_figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.lineup_ax = self.lineup_figure.add_subplot(111)
        self.lineup_ax.plot(self.x_names, self.y_pts)
        self.lineup_graph = FigureCanvasTkAgg(self.lineup_figure, self.lineup_tab)
        self.lineup_graph.get_tk_widget().pack(side='right', fill='both', pady=5, expand=1)

        self.lineup_ax.set_title('Lineup Performance')
        self.lineup_ax.set_ylim(0, 40)
        self.lineup_ax.set_xlim(0, 9)

        self.lineup_ax.tick_params(axis='x', labelrotation=45)

    def update_graph(self):

        lineup = self.lineup_object.get_lineup()

        season = self.test_year.get()
        week = self.week.get()
        self.x_names = list()
        self.y_pts = list()
        fd_pts = list()

        for p in lineup:
            self.x_names.append(p.name)
            self.y_pts.append(p.proj_pts)
            fd_pts.append(p.fd_pts)

        # clear previous axes and plot new one
        self.lineup_ax.cla()
        self.lineup_ax.plot(self.x_names, self.y_pts)
        self.lineup_ax.plot(self.x_names, fd_pts)

        # give plot a line style and marker points
        self.lineup_ax.plot(self.y_pts, linestyle='--', marker='o', color='b')
        self.lineup_ax.plot(fd_pts, linestyle='--', marker='o', color='r')

        self.lineup_ax.set_title("Season: " + f'{season}' + " Week: " + f'{week}')
        self.lineup_ax.tick_params(axis='x', labelrotation=45)

        # allows clicking on marker points to see x and y values
        mplcursors.cursor(self.lineup_ax)

        # tightens up graph so player names don't fall off
        self.lineup_figure.tight_layout()

        # draws the graph
        self.lineup_graph.draw()

    def clear_search(self):

        # deletes the text in the entry box
        self.enter.delete(0, 'end')

        # updates tree by searching for everything
        self.search_player()

    def search_player(self):

        query = """ 
                SELECT player, team, position, season, week, fdsal, nn_score, will_play
                FROM Games
                WHERE player LIKE ?
                AND season = ?
                AND week = ?"""

        text_param = self.text_input.get()
        season = self.test_year.get()
        week = self.week.get()

        # get data from db
        result = self.datafetcher.fetch_query(query, ['%'+text_param+'%', season, week])

        # update the results of the table
        self.update_table(result)

    def update_table(self, query_result):

        self.tree.tag_configure('not playing', background="gray")

        # delete everything in the tree
        self.tree.delete(*self.tree.get_children())

        # highlight all players who won't be playing in gray
        for record in query_result:

            # if will_play is set to 1
            if record[7]:
                self.tree.insert('', 'end', values=(record[0], record[1], record[2], record[3], record[4],
                                                    record[5], record[6]))

            # else highlight gray
            else:
                self.tree.insert('', 'end', values=(record[0], record[1], record[2], record[3], record[4],
                                                    record[5], record[6]), tags=('not playing',))

    def open_player_data(self, event):

        # get tree values
        selection = self.get_tree_selection()

        # vars needed to make a player object
        name = selection[0]
        season = self.test_year.get()
        week = self.week.get()

        # initialize player object
        player_obj = Player(name, season, week)
        player_obj.set_player_data()
        player_display = PlayerDisplay(self.master, player_obj)

    def add_to_lineup(self):

        # returns highlighted row from table
        selection = self.get_tree_selection()

        # vars needed to make player object
        name = selection[0]
        season = self.test_year.get()
        week = self.week.get()

        # initialize Player object and set data
        player_object = Player(name, season, week)
        player_object.set_player_data()

        # if player is scheduled to play
        if player_object.get_will_play():

            # returns True if the player can be added given the constraints
            if self.lineup_object.select_player(player_object):

                # add to the lineup tree
                self.insert_lineup_into_lineup_tree()

    def reset_lineup(self):

        # delete current lineup object
        del self.lineup_object
        season = self.test_year.get()
        week = self.week.get()

        # re-initialize a new one
        self.lineup_object = Lineup(season, week)

        # delete lineup tree
        self.lineup_tree.delete(*self.lineup_tree.get_children())

    def remove_from_lineup(self):

        # get data from lineup tree
        selection = self.lineup_tree.selection()[0]  # returns memory location
        name = self.lineup_tree.item(selection, "values")  # gets actual value: ('name',)
        season = self.test_year.get()
        week = self.week.get()

        # create player object
        player = Player(name[0], season, week)
        player.set_player_data()

        # remove from lineup object
        self.lineup_object.remove_player(player)

        # insert the lineup again but without the player
        self.insert_lineup_into_lineup_tree()

    def schedule_to_play(self):

        # returns highlighted row from table
        selection = self.get_tree_selection()

        # vars needed to make player object
        name = selection[0]
        season = self.test_year.get()
        week = self.week.get()

        # initialize object
        player_object = Player(name, season, week)

        # sets appropriate data
        player_object.set_player_data()
        player_object.set_will_play_true()

        # update player pool
        self.lineup_object.update_pool()

        # updates tree
        self.search_player()

    def unschedule_from_play(self):

        selection = self.get_tree_selection()

        # vars needed to make player object
        name = selection[0]
        season = self.test_year.get()
        week = self.week.get()

        # initialize object
        player_object = Player(name, season, week)

        # sets appropriate data
        player_object.set_player_data()
        player_object.set_will_play_false()

        # update player pool
        self.lineup_object.update_pool()

        # updates tree
        self.search_player()

    def train_players(self):

        season = self.test_year.get()

        # initialize NNs for each position
        qb_nn = NeuralNet(self.master, 'QB', season)
        wr_nn = NeuralNet(self.master, 'WR', season)
        rb_nn = NeuralNet(self.master, 'RB', season)
        te_nn = NeuralNet(self.master, 'TE', season)
        def_nn = NeuralNet(self.master, 'DEF', season)

        # train NNs for each position then show progress
        qb_nn.train()
        wr_nn.train()
        rb_nn.train()
        te_nn.train()
        def_nn.train()

        # refresh table
        self.search_player()

        # OK messagebox showing that training has been completed for every position
        tk.messagebox.showinfo(title=None, message="Training complete.")

    def get_tree_selection(self):

        selection = self.tree.selection()
        values = []
        for i in self.tree.item(selection, "values"):
            values.append(i)

        return values

    def generate_lineup(self):

        self.lineup_object.generate_lineup()
        self.insert_lineup_into_lineup_tree()

    def insert_lineup_into_lineup_tree(self):

        # get lineup
        lineup = self.lineup_object.get_lineup()

        # delete everything that's in lineup already
        self.lineup_tree.delete(*self.lineup_tree.get_children())

        # for each player in lineup, insert into lineup tree
        for p in lineup:

            # insert parent
            parent = self.lineup_tree.insert('', 0, text='Player', values=(p.name,), open=False)

            # add children
            child1 = self.lineup_tree.insert(parent, 'end', text='Child1', values=("Proj Pts: " + str(p.proj_pts),))
            child2 = self.lineup_tree.insert(parent, 'end', text='Child2', values=("Salary: " + str(p.fd_sal),))
            child3 = self.lineup_tree.insert(parent, 'end', text='Child3', values=("Position: " + p.position,))
            child4 = self.lineup_tree.insert(parent, 'end', text='Child4', values=("Team: " + p.team,))

        # re-adjust totals
        self.insert_totals_into_lineup_tree()

    def insert_totals_into_lineup_tree(self):

        total_sal = self.lineup_object.get_total_salary()
        total_pts = np.round(self.lineup_object.get_total_points(), 3)

        self.lineup_tree.insert('', 'end', text='Salary', values=("Total Salary: " + str(total_sal),))
        self.lineup_tree.insert('', 'end', text='Points', values=("Total Pts: " + str(total_pts),))

    def open_nn_parameters(self):

        frame = tk.Frame(self.master)
        nn_window = tk.Toplevel(frame)
        season = self.test_year.get()

        nn_window.title("NN Parameters: " + f'{season}')
        nn_window.geometry('400x200')

        canvas1 = tk.Canvas(nn_window, bd=1, relief=tk.SUNKEN)
        canvas1.pack(fill='both', expand=1)

    def create_new_by_file(self):

        answer = tk.messagebox.askquestion('New', 'Do you want to start over?')

        if answer == 'yes':

            conn = db.connect('data.db')
            curs = conn.cursor()

            games_df = pd.read_csv('FD_data_all.csv')
            list_of_players = games_df['player'].values

            # removes duplicate names
            list_of_players = list(dict.fromkeys(list_of_players))
            
            # lists to be written to db containing all values for all players
            all_running_avgs = []
            all_variances = []

            for player in list_of_players:

                # temp lists for moving avg and variance
                sma = []
                variances = []
                
                # temp variable
                pts = 0

                point_list = games_df[games_df.player == f'{player}'].fd_points

                # resets index of dataframe to 0
                point_list = point_list.reset_index(drop=True)

                # interpolated average to be used as first input for player's moving avg
                temp_avg = np.round(point_list.mean(), 3)
                sma.append(temp_avg)

                # standard deviation needed for variance
                std = np.nanstd(point_list)
                variances.append((std/temp_avg) * 100)

                # create avg
                for j in range(len(point_list) - 1):

                    total = j + 1
                    # creates a simple moving average.
                    pts += point_list[j]
                    avg = pts / total
                    avg = np.round(avg, 3)
                    sma.append(avg)
                    
                    # create variance: var = std/mean
                    std = np.nanstd(sma)
                    # when the value of the std is 0 (which it will be once for every new player)
                    # just give it the variance for their whole career up to this point
                    if std == 0:

                        variances.append(variances[0])

                    # TODO find a way to get rid of this
                    avg += .0001
                    var = np.round((std/avg) * 100, 2)
                    variances.append(var)

                # copy running avgs/vars of each player to one big list
                for i in range(len(sma)):

                    all_running_avgs.append(sma[i])
                    all_variances.append(variances[i])

            # put all running avgs/variances in new dataframe column
            games_df.insert(loc=0, column='avg_pts', value=all_running_avgs)
            games_df.insert(loc=0, column='variances', value=all_variances)

            # write dataframe to sqlite db
            engine = create_engine('sqlite:///data.db', echo=False)
            engine.execute("DROP table IF EXISTS Games")
            games_df.to_sql('Games', con=engine)
            engine.execute("SELECT * FROM Games").fetchall()
            conn.commit()
            conn.close()

    def open_test_window(self, playerclass):
        print("callback")

        new_window = playerclass(self.master)

    def export_db_to_csvfile(self, csv_file):

        pass

    def write_to_database(self, player_object):

        # if this is the current week, then the fanduel pts aren't present
        if player_object.fd_pts is None:

            insert_query = """
                    INSERT INTO Games
                    (player, position, team, season, week, fdsal, dksal, opp, nn_score, avg_pts, will_play)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

            inputs = [player_object.name,
                      player_object.position,
                      player_object.team,
                      player_object.season,
                      player_object.week,
                      player_object.fd_sal,
                      player_object.dk_sal,
                      player_object.opponent,
                      player_object.proj_pts,
                      player_object.season_avg]

            # if the player's salaries are greater than 0, he can potentially play
            # add this to the input list after you've made change
            if player_object.fd_sal > 0 and player_object.dk_sal > 0:
                player_object.will_play = 1
                inputs.append(player_object.will_play)

            else:
                player_object.will_play = 0
                inputs.append(player_object.will_play)

            self.datafetcher.execute_query(insert_query, inputs)

        # if the fanduel pts are present then this is a prior week
        # and we need to update that field in the database as well as the final
        # salaries
        elif player_object.fd_pts is not None:

            insert_query = """
                        INSERT INTO Games
                        (player, season, week, fdsal, dksal, fd_points)
                        VALUES (?, ?, ?, ?, ?, ?)"""

            inputs = [player_object.name,
                      player_object.season,
                      player_object.week,
                      player_object.fd_sal,
                      player_object.dk_sal,
                      player_object.fd_pts]

            self.datafetcher.execute_query(insert_query, inputs)

    def update_previous_weeks_data(self):
        """
        updates the fanduel points to what was actually scored last week,and the fanduel salary
        and the draftkings salary to reflect the final salaries. This is important because
        fanduel and draftking salaries may fluctuate throughout the week depending on injury
        reports and other factors. Salary data is needed for any future training, since it is
        part of the input parameters for the NN, and we want to get what salaries were actually
        settled on.
        """
        season = self.test_year.get()
        week = self.week.get()

        # change week to last week
        prev_week = str(int(week) - 1)

        # initialize scraper and get all the data from last week
        scraper = WebScraper(season, prev_week)

        # dictionary: {player: [position, team, opposition, fanduel pts, fd salary, dk salary]
        player_dict = scraper.get_all_historical_data()

        for key in player_dict:

            name = key

            # initialize player object and set scraped values to it
            player_object = Player(name, season, prev_week)

            # currently these are the only values we need from the dictionary
            player_object.fd_pts = player_dict[key][3]
            player_object.fd_sal = player_dict[key][4]
            player_object.dk_sal = player_dict[key][5]

            # write these updated values to the db
            self.write_to_database(player_object)

            # memory management
            del player_object

    def update_player_data(self):
        """
        Updates the current season and week with data from rotowire,
        then retrieves inputs needed for NN, evaluates inputs with NN to get projected pts, and finally
        makes updates to database.
        """
        season = self.test_year.get()
        week = self.week.get()

        # ensures last weeks fanduel points are updated
        self.update_previous_weeks_data()

        # initialize scraper for current season/week
        scraper = WebScraper(season, week)

        # returns dict of necessary data for the upcoming games
        # dictionary - {player: position, team, opponent, fd sal, dk sal}
        player_dict = scraper.get_all_current_data()

        # initializes neural net objects for each position
        qb_nn = NeuralNet(self.master, 'QB', season)
        rb_nn = NeuralNet(self.master, 'RB', season)
        wr_nn = NeuralNet(self.master, 'WR', season)
        te_nn = NeuralNet(self.master, 'TE', season)
        def_nn = NeuralNet(self.master, 'DEF', season)

        # load the saved node weights for each nn that was obtained through training
        qb_nn.load_saved_weights()
        rb_nn.load_saved_weights()
        wr_nn.load_saved_weights()
        te_nn.load_saved_weights()
        def_nn.load_saved_weights()

        # create player objects from the data in the dictionary
        for key in player_dict:
            name = key
            player_object = Player(name, season, week)

            # set values from dictionary to object
            player_object.position = player_dict[name][0]
            player_object.team = player_dict[name][1]
            player_object.opponent = player_dict[name][2]
            player_object.fd_sal = player_dict[name][3]
            player_object.dk_sal = player_dict[name][4]

            # calculate current season avg for NN input - this is one of the inputs
            # other stats may also be calculated for other uses
            player_object.calculate_season_stats()

            # inputs to be used in order to get projected pts
            inputs = [player_object.fd_sal, player_object.dk_sal, player_object.season_avg]

            # get projected points for each position
            if player_object.position == 'QB':
                player_object.proj_pts = qb_nn.evaluate_player(inputs)

            elif player_object.position == 'RB':
                player_object.proj_pts = rb_nn.evaluate_player(inputs)

            elif player_object.position == 'WR':
                player_object.proj_pts = wr_nn.evaluate_player(inputs)

            elif player_object.position == 'TE':
                player_object.proj_pts = te_nn.evaluate_player(inputs)

            elif player_object.position == 'DEF':
                player_object.proj_pts = def_nn.evaluate_player(inputs)

            # write the data contained in player object to db
            self.write_to_database(player_object)

            # memory management
            del player_object

    def calculate_player_stats(self):

        seasons = [int(i) for i in range(2015, 2021)]
        weeks = [int(i) for i in range(1, 18)]

        progress_length = len(seasons) * len(weeks)
        progress_bar = Progress(self.master, 'Calculating stats', progress_length)

        for season in seasons:

            for week in weeks:

                # get all the current players this week
                query = 'SELECT player FROM Games WHERE season = ? AND week = ? AND will_play = ?'

                inputs = (season, week, 1, )  # must be a tuple

                # returns a dataframe with only the players who've played and were scheduled to play
                df = self.datafetcher.get_dataframe(query, inputs)
                players = df['player'].values

                # to be used for execute many query
                update_lists = list()

                for player in players:

                    query = """
                            SELECT fd_points, fdsal, dksal
                            FROM Games
                            WHERE player = ?
                            AND season = ?
                            AND week < ?
                            AND will_play = ?"""

                    # create a dataframe
                    inputs = [player, season, week, 1]
                    df = self.datafetcher.get_dataframe(query, inputs)

                    # if the dataframe is empty then there are no previous games to get the avg for,
                    # so just use the current game as the starting point
                    if df.empty:

                        query = """
                                SELECT fd_points, fdsal, dksal
                                FROM Games
                                WHERE player = ?
                                AND season = ?
                                AND week = ?
                                AND will_play = ?"""

                        df = self.datafetcher.get_dataframe(query, inputs)

                    # calculate average stats
                    avg_pts = df['fd_points'].mean()
                    avg_fdsal = df['fdsal'].mean()
                    avg_dksal = df['dksal'].mean()

                    # store in one large list in order to execute all at once
                    temp_list = [avg_pts, avg_fdsal, avg_dksal, player, season, week]
                    update_lists.append(temp_list)

                # increment progress bar
                progress_bar.update_progress(1)

                update_query = """
                               UPDATE Games 
                               SET avg_pts = ?, 
                               avg_fdsal = ?,
                               avg_dksal = ?
                               WHERE player = ?
                               AND season = ?
                               AND week = ?"""

                # update player stats
                self.datafetcher.execute_many(update_query, update_lists)

    def create_new_by_scraping(self):

        # drop the Games table in database
        self.datafetcher.drop_table('Games')

        # create a new games table with no data
        self.datafetcher.create_games_table()

        seasons = [int(i) for i in range(2015, 2021)]
        weeks = [int(i) for i in range(1, 18)]

        progress_length = len(seasons) * len(weeks)
        progress_bar = Progress(self.master, 'Scraping web data', progress_length)

        insert_list = list()

        for season in seasons:

            for week in weeks:

                webscraper = WebScraper(season, week)

                # dictionary --> {player: [position, team, opponent, fd_pts, fd_sal, dk_sal]}
                player_dict = webscraper.get_all_historical_data()

                print(player_dict)

                for name in player_dict:

                    fd_pts = float(player_dict[name][3])
                    fd_sal = float(player_dict[name][4])
                    dk_sal = float(player_dict[name][5])

                    # set will_play bit based on pts and salaries
                    if fd_pts != 0 and fd_sal > 0 and dk_sal > 0:
                        will_play = 1
                    else:
                        will_play = 0

                    inputs = (name,
                              season,
                              week,
                              player_dict[name][0],  # position
                              player_dict[name][1],  # team
                              player_dict[name][2],  # opponent
                              fd_pts,
                              fd_sal,
                              dk_sal,
                              0,                     # nn_score
                              0,                     # avg_pts
                              will_play,
                              0,                     # avg_fdsal
                              0,)                    # avg_dksal

                    insert_list.append(inputs)

                    """determine if the player actually played that game
                    this is needed because we don't want to go on to train the neural net
                    with players who didn't play but were scheduled to play
                    or with players who weren't scheduled to play ($0 salaries) but did play.
                    This introduces too much noise into the training set"""

                # update progress status
                progress_bar.update_progress(1)

                del webscraper

        insert_query = """
                        INSERT INTO Games
                        (player, season, week, position, team, opp, fd_points, 
                        fdsal, dksal, nn_score, avg_pts, will_play, avg_fdsal, avg_dksal)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        self.datafetcher.execute_many(insert_query, insert_list)

        # get the data that will be used for nn inputs
        self.calculate_player_stats()

def main():
    root = tk.Tk()
    application = MainDisplay(root)
    root.mainloop()


if __name__ == '__main__':
    main()



