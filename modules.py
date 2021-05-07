import sqlite3 as db
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools
from itertools import zip_longest
import pulp
import bs4 as bs
import urllib3
import requests
import re
from selenium import webdriver

########################################################################################################################
class DataFetcher:

    def __init__(self):

        self.database = 'data.db'
        self.conn = None
        self.curs = None

    def open_connection(self):

        try:

            self.conn = db.connect(self.database)
            self.curs = self.conn.cursor()

        except db.Error:

            print('connection error')

    def execute_query(self, query, inputs=None):

        self.open_connection()

        self.curs.execute(query, inputs)
        self.conn.commit()

        self.close_connection()

    def execute_many(self, query, inputs=None):

        self.open_connection()

        self.curs.executemany(query, inputs)
        self.conn.commit()

        self.close_connection()

    def fetch_query(self, query, inputs):

        self.open_connection()

        result = self.curs.execute(query, inputs).fetchall()
        self.conn.commit()

        self.close_connection()

        return result

    def get_dataframe(self, query, inputs):

        self.open_connection()

        df = pd.read_sql_query(query, self.conn, params=inputs)

        self.close_connection()

        return df

    def get_dataframe_filtered(self, query, inputs):

        self.open_connection()

        # df =

        self.close_connection()

        # return df

    def create_table(self, query, table_name):

        self.open_connection()

        self.close_connection()

    def drop_table(self, table_name):

        self.open_connection()

        self.curs.execute("DROP TABLE IF EXISTS '{table}'".format(table=table_name))
        self.conn.commit()

        self.close_connection()

    def create_games_table(self):

        query = """
                CREATE TABLE IF NOT EXISTS Games
                (id INTEGER PRIMARY KEY,
                player TEXT,
                season INTEGER,
                week INTEGER,
                team TEXT,
                position TEXT,
                opp TEXT,
                fd_points FLOAT,
                fdsal BIGINT,
                dksal BIGINT,
                nn_score FLOAT,
                will_play INTEGER,
                avg_pts FLOAT,
                avg_fdsal FLOAT,
                avg_dksal FLOAT)
                """

        self.execute_query(query, [])

    def close_connection(self):

        self.curs.close()
        self.conn.close()

########################################################################################################################


class Progress:

    def __init__(self, master, progress_type, length):

        # window variables
        self.master = master
        self.frame = tk.Frame(self.master)
        self.p_bar_window = tk.Toplevel(self.frame)
        self.p_bar_window.title(f'{progress_type}')

        # dimensions for centering
        main_width = self.master.winfo_screenwidth()
        main_height = self.master.winfo_screenheight()

        # vars for centering
        window_width = 800
        window_height = 25
        half_main_width = int(main_width/2 - window_width/2)
        half_main_height = int(main_height/2 - window_height/2)

        # center window in middle of app
        self.p_bar_window.geometry(f'{window_width}x{window_height}+{half_main_width}+{half_main_height}')

        # maximum length of progress
        self.length = length

        # initialize progress bar and display it
        self.progress = ttk.Progressbar(self.p_bar_window, orient=tk.HORIZONTAL, length=self.length, mode='determinate')
        self.progress.pack(fill='both', expand=1)

        # sets maximum length of progress bar
        self.progress['maximum'] = self.length

        # pauses main program to launch progress bar -- i.e., the idle task in this instance
        self.master.update_idletasks()

    def update_progress(self, step_length):

        # pauses main program to update the progress bar
        self.master.update_idletasks()

        # increments the value of progress bar
        self.progress.step(step_length)

        # updates progress bar widget
        self.progress.update()

        self.master.update_idletasks()
        time.sleep(.001)

        # destroy window if maximum progress has been reached
        if self.progress['value'] == self.length:
            self.progress.stop()
            self.p_bar_window.destroy()

########################################################################################################################

class NeuralNet:

    def __init__(self, master, position, test_year):

        self.master = master
        self.position = position
        self.test_year = test_year
        self.datafetcher = DataFetcher()

        # to be modified into parameters later
        self.learning_rate = .1
        # sigmoid activation function
        self.activation_fx = lambda x: sp.special.expit(x)
        self.input_nodes = 3
        self.hidden_nodes = 10
        self.output_nodes = 1
        self.epochs = 5
        self.epoch_error_rate = list()

        # progress bar variables
        self.progress_bar = None
        self.training_length = 0
        self.test_length = 0

        # dataframe variables for min-max normalization
        self.avg_fdsal_min = 0
        self.avg_fdsal_max = 0
        self.avg_dksal_min = 0
        self.avg_dksal_max = 0
        self.avg_pts_min = 0
        self.avg_pts_max = 0
        self.std_max = 0
        self.fdpts_min = 0
        self.fdpts_max = 0

        # standard normal initialization
        self.input_hidden_link = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                  (self.hidden_nodes, self.input_nodes))

        self.hidden_output_link = np.random.normal(0.0, pow(self.output_nodes, -0.5),
                                                   (self.output_nodes, self.hidden_nodes))
        # initialize dataframe variables
        self.set_vars()

    def train(self):

        # opens separate window with progress bar placed in it
        self.initialize_progress_bar()

        # declare error rate variable
        avg_error_rate = float()

        # one epoch is one pass through of all the data
        while self.epochs > 0:

            # returns results shuffled randomly to facilitate improved training
            result = self.get_randomized_player_data()

            # used for calculating the average error rate for each epoch
            error_rate = float()
            total = 1

            for record in result:

                # reassign inputs for clarity
                avg_fdsal = record[0]
                avg_dksal = record[1]
                avg_pts = record[2]

                # target - this is what we want to try and predict through training
                fd_pts = record[3]

                # inputs for nn - normalize them first
                input_list = self.normalize_inputs([avg_fdsal, avg_dksal, avg_pts])

                # value to be predicted/error correction
                target_list = self.normalize_target(fd_pts)

                # transpose into array with two dimensions
                inputs = np.array(input_list, ndmin=2).T
                target = np.array(target_list, ndmin=2).T

                # get dot product of inputs
                hidden_inputs = np.dot(self.input_hidden_link, inputs)

                # activation function affects values going into next layer
                hidden_outputs = self.activation_fx(hidden_inputs)

                # get dot product of hidden inputs then run them through activation function
                final_inputs = np.dot(self.hidden_output_link, hidden_outputs)
                final_output = self.activation_fx(final_inputs)

                # difference between what we want and what the nn currently gave us
                output_error = target - final_output

                # this will be used for graphing how well training is working
                error_rate += output_error/final_output
                avg_error_rate = error_rate/total
                total += 1

                hidden_errors = np.dot(self.hidden_output_link.T, output_error)

                # updates the input to hidden layer and the hidden to output layer links
                # the partial derivative of the logistic function is the output multiplied by 1 minus the output
                self.hidden_output_link += self.learning_rate * np.dot((output_error * final_output
                                                                        * (1.0 - final_output)), np.transpose(hidden_outputs))

                self.input_hidden_link += self.learning_rate * np.dot((hidden_errors * hidden_outputs
                                                                        * (1.0 - hidden_outputs)), np.transpose(inputs))
                self.progress_bar.update_progress(1)

            self.epochs -= 1

            # add whatever the last avg error rate was for the epoch
            self.epoch_error_rate.append(avg_error_rate)

        # score every player in test year (season) and save the input and output node weights
        self.score_players()
        self.save_nn()

    def initialize_progress_bar(self):

        # set progress bar length
        progress_length = (self.training_length * self.epochs) + self.test_length

        # initialize progress object
        self.progress_bar = Progress(self.master, f'{self.position}' + ' Training', progress_length)

    def get_avg_error_rates(self):
        """
        returns list of the average error rate recorded at the end of each epoch
        index 0 is the first epoch, index 1 is the second...and so forth.
        """
        return self.epoch_error_rate

    def get_randomized_player_data(self):

        query = """
                SELECT avg_fdsal, avg_dksal, avg_pts, fd_points 
                FROM Games 
                WHERE position = ? 
                AND season != ?
                AND will_play = 1 
                ORDER BY RANDOM()"""

        result = self.datafetcher.fetch_query(query, [self.position, self.test_year])

        return result

    def score_players(self):

        query = """
                SELECT ROWID, player, avg_fdsal, avg_dksal, avg_pts 
                FROM Games
                WHERE season = ?
                AND position = ?
                AND will_play = 1"""

        result = self.datafetcher.fetch_query(query, [self.test_year, self.position])

        for record in result:

            # declare new variables for clarity
            row_id = record[0]
            avg_fdsal = record[2]
            avg_dksal = record[3]
            avg_pts = record[4]

            # get player projected points
            nn_value = self.evaluate_player([avg_fdsal, avg_dksal, avg_pts])

            # write projected score to db
            update_query = """
                           UPDATE Games
                           SET nn_score = ?
                           WHERE ROWID = ?"""

            self.datafetcher.execute_query(update_query, [nn_value, row_id])

            self.progress_bar.update_progress(1)

    def set_vars(self):

        # create dataframes
        temp_df = self.datafetcher.get_dataframe("SELECT * FROM Games", [])

        # filter dataframes further
        df = temp_df[(temp_df['season'] != int(self.test_year)) & (temp_df['position'] == self.position) &
                     (temp_df['will_play'] == 1)]

        # this is used exclusively for assisting in size of progress bar
        df2 = temp_df[(temp_df['season'] == int(self.test_year)) & (temp_df['position'] == self.position) &
                      (temp_df['will_play'] == 1)]

        # used to help determine size of progress bar
        self.training_length = df.shape[0]
        self.test_length = df2.shape[0]

        # min-max normalization variables
        self.avg_fdsal_min = df['avg_fdsal'].min()
        self.avg_fdsal_max = df['avg_fdsal'].max()
        self.avg_dksal_min = df['avg_dksal'].min()
        self.avg_dksal_max = df['avg_dksal'].max()
        self.avg_pts_min = df['avg_pts'].min()
        self.avg_pts_max = df['avg_pts'].max()
        self.fdpts_min = df['fd_points'].min()
        self.fdpts_max = df['fd_points'].max()

    def evaluate_player(self, input_list):
        """
        this function takes a list of three input parameters, normalizes them,
        evalutes them with the neural network based off the trained node weights,
        denormalizes the output and then finally returns a floating point value.
        :param [fanduel salary, dradftking salary, average season fanduel points]:
        :return: float - projected fanduel points for the upcoming week
        """

        # normalize input list
        normed_inputs = self.normalize_inputs(input_list)

        inputs = np.array(normed_inputs, ndmin=2).T
        hidden_inputs = np.dot(self.input_hidden_link, inputs)

        hidden_outputs = self.activation_fx(hidden_inputs)

        final_inputs = np.dot(self.hidden_output_link, hidden_outputs)
        final_output = self.activation_fx(final_inputs)

        denormed_final = self.denormalize_output(final_output)

        return denormed_final

    def normalize_inputs(self, input_list):

        normed_sal1 = (input_list[0] - self.avg_fdsal_min) / (self.avg_fdsal_max - self.avg_fdsal_min)
        normed_sal2 = (input_list[1] - self.avg_dksal_min) / (self.avg_dksal_max - self.avg_dksal_min)
        normed_avg_pts = (input_list[2] - self.avg_pts_min) / (self.avg_pts_max - self.avg_pts_min)

        return [normed_sal1, normed_sal2, normed_avg_pts]

    def normalize_target(self, fd_pt_input):

        normed_fd_pts = (fd_pt_input - self.fdpts_min) / (self.fdpts_max - self.fdpts_min)

        return [normed_fd_pts]

    def denormalize_output(self, output):

        # reverses min-max normalization through algebra
        out_var = output[0][0]
        denormed_output = out_var * (self.fdpts_max - self.fdpts_min) + self.fdpts_min

        # round to three decimal places
        denormed_output = np.round(denormed_output, 3)

        return denormed_output

    def save_nn(self):

        conn = db.connect('data.db', timeout=30)
        curs = conn.cursor()

        # make numpy array into list of sublists
        input_link_list = self.input_hidden_link.tolist()
        output_link_list = self.hidden_output_link.tolist()

        # make sublists into one big list in order to write to db
        input_link_list = list(itertools.chain.from_iterable(input_link_list))
        output_link_list = list(itertools.chain.from_iterable(output_link_list))

        # name the table
        table_name = self.position + "_" + self.test_year + "_nn_weights"
        # drop any previous table
        query = """
                DROP TABLE IF EXISTS '{table}'
                """.format(table=table_name)
        curs.execute(query)
        conn.commit()

        # create new NN table of year and position for input and output weights
        table_query = """
                CREATE TABLE '{table}'
                (id INTEGER PRIMARY KEY, 
                i_weights FLOAT,
                o_weights FLOAT)""".format(table=table_name)
        curs.execute(table_query)
        conn.commit()
        temp = list()
        insert_tuple_list = []
        # make a tuple of two values
        for i, j in zip_longest(input_link_list, output_link_list):
            temp = [i, j]
            temp = tuple(temp)
            insert_tuple_list.append(temp)

        insert_query = """
                INSERT INTO '{table}'
                (i_weights, o_weights)
                VALUES (?, ?)""".format(table=table_name)
        curs.executemany(insert_query, insert_tuple_list)
        conn.commit()

    def load_saved_weights(self):

        conn = db.connect('data.db')
        curs = conn.cursor()

        table_name = self.position + "_" + self.test_year + "_nn_weights"

        query = """
                SELECT i_weight, o_weights
                FROM '{table}'""".format(table=table_name)

        result = curs.execute(query).fetchall()
        conn.commit()

        input_weights = list()
        output_weights = list()
        temp = list()

        # reformat weights so they can be converted from lists back to arrays
        for weight in result:

            # need to create a list of lists of 3
            temp.append(weight[0])

            if len(temp) == self.input_nodes:
                input_weights.append(temp)
                temp = list()

            # there's more input nodes than output nodes so we need to discard None entries
            # this is a byproduct of saving unequal columns to db
            if weight[1] is not None:
                output_weights.append(weight[1])

        # reset weights
        self.input_hidden_link = np.array(input_weights)
        self.hidden_output_link = np.array(output_weights)

        conn.close()

    def set_learning_rate(self, lr):

        self.learning_rate = lr

########################################################################################################################

class Player:

    def __init__(self, name, season, week):

        # fetches/executes data via db
        self.datafetcher = DataFetcher()

        self.name = name
        self.season = season
        self.week = week
        self.team = None
        self.opponent = None
        self.position = None
        self.fd_sal = None
        self.dk_sal = None
        self.season_avg = None
        self.proj_pts = None
        self.fd_pts = None
        self.will_play = None
        self.season_var = None

    def set_player_data(self):

        query = """
                SELECT team, position, fdsal, nn_score, fd_points, will_play
                FROM Games
                WHERE player = ?
                AND season = ?
                AND week = ?"""

        result = self.datafetcher.fetch_query(query, [self.name, self.season, self.week])

        # set the data
        self.team = result[0][0]
        self.position = result[0][1]
        self.fd_sal = result[0][2]
        self.proj_pts = result[0][3]
        self.fd_pts = result[0][4]
        self.will_play = result[0][5]

    def calculate_season_stats(self):

        # dataframe query
        query = 'SELECT * FROM Games'

        # if this is the first game played this season,
        if self.week == 1:

            #  get the avg of previous seasons up to now for the first value
            df = self.datafetcher.get_dataframe(query, [])
            df = df[(df['player'] == self.name) &
                    (df['season'] <= int(self.season)) &
                    (df['week'] < int(self.week))]

            self.season_avg = df['fd_points'].mean()
            standard_dev = df['fd_points'].std()
            self.season_var = standard_dev/self.season_avg

        else:

            # otherwise, just get the season avg up to now
            df = self.datafetcher.get_dataframe(query, [])
            df = df[(df['player'] == self.name) &
                (df['season'] == int(self.season)) &
                (df['week'] < int(self.week))]

            self.season_avg = df['fd_points'].mean()
            standard_dev = df['fd_points'].std()
            self.season_var = standard_dev / self.season_avg

    def get_will_play(self):

        return self.will_play

    def set_will_play_true(self):

        query = """
                UPDATE Games
                SET will_play = 1
                WHERE player = ?
                AND season = ?
                AND week = ?"""

        self.datafetcher.execute_query(query, [self.name, self.season, self.week])

    def set_will_play_false(self):

        query = """
                UPDATE Games
                SET will_play = 0
                WHERE player = ?
                AND season = ?
                AND week = ?"""

        self.datafetcher.execute_query(query, [self.name, self.season, self.week])

    def __eq__(self, other):

        return self.name == other.name

########################################################################################################################

class WebScraper:

    def __init__(self, season, week):

        self.season = season
        self.week = week

        # used to login to rotowire to scrape current players and their salaries
        self.username = 'baw5xc'
        self.password = 'Xpw9000*'

        # stats_dict will be nested here
        self.player_data = list()
        self.name_dict = dict()

        # log in to rotowire
        # self.log_into_rotowire()

    def log_into_rotowire(self):

        driver = webdriver.Chrome()
        driver.get('https://www.rotowire.com/users/login.php?go=%2F')

        login_username = driver.find_element_by_name('username')
        login_username.send_keys(self.username)

        login_password = driver.find_element_by_name('password')
        login_password.send_keys(self.password)

        # login button
        driver.find_element_by_name('button').click()
        driver.quit()

    def get_fanduel_source(self):

        driver = webdriver.Chrome()
        # TODO: this needs to match season/week
        driver.get('https://www.rotowire.com/daily/nfl/value-report.php?site=Fanball')
        time.sleep(15)
        page = driver.page_source
        driver.quit()

        return page

    def get_draftkings_source(self):

        driver = webdriver.Chrome()

        # TODO: this needs to match season/week
        driver.get('https://www.rotowire.com/daily/nfl/value-report.php?site=Head2Head')
        time.sleep(15)
        page = driver.page_source
        driver.quit()

        return page

    def scrape_historical_fd_data(self):

        url = "http://rotoguru1.com/cgi-bin/fyday.pl?week={}&year={}&game=fd&scsv=1".format(self.week, self.season)
        page = requests.get(url)

        soup = bs.BeautifulSoup(page.text, 'html.parser')
        data = soup.find('pre').text

        data_list = data.splitlines()

        # final list of lists of formatted data
        formatted_data = list()

        for item in data_list:

            temp_data = list()
            string = ''
            for s in item:

                if s == ';':
                    temp_data.append(string)
                    string = ''
                else:
                    string += s

            # append the last string (i.e., salary)
            if string == '':

                # if value is N/A (null), then append 0
                temp_data.append(0)

            else:
                temp_data.append(string)

            # make list of lists
            formatted_data.append(temp_data)

        # remove the first element, which are the column names
        del formatted_data[0]

        # extract fd data for players
        for player in formatted_data:

            # if the position is PK, continue loop b/c position is obsolete and not used in all seasons
            if player[4] == 'PK':

                continue

            # format name to 'first name last name', instead of 'last name, first name'
            lastname_firstname = player[3]

            # not all names are formatted last name, first name
            # the defenses don't have commas in their names
            comma_index = lastname_firstname.index(',') if ',' in lastname_firstname else None

            # if there's a comma, then change format to first name last name
            if comma_index:

                first_name = lastname_firstname[comma_index + 2:]
                last_name = lastname_firstname[:comma_index]
                name = first_name + ' ' + last_name

            # otherwise it's a defense, so keep it the same and store var
            else:

                name = player[3]

            # temporary list
            tl = list()

            tl.append(str.upper(player[4]))  # position
            tl.append(player[5])             # team
            tl.append(player[7])             # opp
            tl.append(player[8])             # fd_pts
            tl.append(player[9])             # fd_sal

            # link player to his stats
            self.name_dict[name] = tl

    def scrape_historical_dk_salary(self):

        url = "http://rotoguru1.com/cgi-bin/fyday.pl?week={}&year={}&game=dk&scsv=1".format(self.week, self.season)
        page = requests.get(url)

        soup = bs.BeautifulSoup(page.text, 'html.parser')
        data = soup.find('pre').text

        data_list = data.splitlines()

        # format data into lists
        formatted_data = []

        for item in data_list:

            string = ''
            temp_data = []
            for s in item:

                if s == ';':
                    temp_data.append(string)
                    string = ''
                else:
                    string += s

            # append last string
            temp_data.append(string)

            # put it all in list of lists
            formatted_data.append(temp_data)

        # remove the first element
        del formatted_data[0]

        # temporary list to store all dk players and their salaries
        temp_names = list()
        temp_sal = list()

        # extract dk salary for players
        for player in formatted_data:

            # format name to 'first name last name', instead of 'last name, first name'
            lastname_firstname = player[3]

            # not all names are formatted last name, first name
            # the defenses don't have commas in their names
            comma_index = lastname_firstname.index(',') if ',' in lastname_firstname else None

            if comma_index:

                first_name = lastname_firstname[comma_index + 2:]
                last_name = lastname_firstname[:comma_index]
                name = first_name + ' ' + last_name

            else:

                name = player[3]

            temp_names.append(name)

            # in case there are null values for dksalaries
            if player[9] == '':
                temp_sal.append(0)

            else:
                temp_sal.append(player[9])  # dk sal

        # look through the fanduel players and see if those players are in draftkings
        for key in self.name_dict:

            # if they are then append their draftking's salary
            if key in temp_names:

                index = temp_names.index(key)
                self.name_dict[key].append(temp_sal[index])

            # otherwise just append a salary of 0
            else:

                self.name_dict[key].append(temp_sal[0])


    def scrape_current_fd_data(self):

        # position, team, opp, salary
        names = list()
        positions = list()
        teams = list()
        opps = list()
        salaries = list()

        page = self.get_fanduel_source()
        soup = bs.BeautifulSoup(page, 'lxml')

        left = soup.find('div', {'class': 'webix_ss_left'})
        for divs in left.find_all('div', {'class': 'webix_cell'}):

            # trim name
            name = divs.text
            index = name.index('.')
            name = name[0:index-1] + name[index+1:]
            names.append(name)

        center = soup.find('div', {'class': 'webix_ss_center'})
        pos = center.find('div', {'column': '2'})
        for p in pos.find_all('div', {'class': 'webix_cell'}):
            positions.append(p.text)

        # team column
        team_column = center.find('div', {'column': '3'})
        for t in team_column.find_all('div', {'class': 'webix_cell'}):
            teams.append(t.text)

        # opponent column
        opp_column = center.find('div', {'column': '4'})
        for o in opp_column.find_all('div', {'class': 'webix_cell'}):
            opps.append(o.text)

        # salary column
        sal_column = center.find('div', {'column': '10'})
        for s in sal_column.find_all('div', {'class': 'webix_cell'}):

            fd_sal = s.text
            # trim dollar sign
            fd_sal = fd_sal[1:]
            salaries.append(fd_sal)

        for i in range(len(names)):

            temp = [positions[i], teams[i], opps[i], salaries[i]]
            self.name_dict[names[i]] = temp

    def scrape_current_dk_salary(self):

        names = list()
        salaries = list()

        page = self.get_draftkings_source()
        soup = bs.BeautifulSoup(page, 'lxml')

        left = soup.find('div', {'class': 'webix_ss_left'})
        for divs in left.find_all('div', {'class': 'webix_cell'}):
            # trim name
            name = divs.text
            index = name.index('.')
            name = name[0:index - 1] + name[index + 1:]
            names.append(name)

        center = soup.find('div', {'class': 'webix_ss_center'})
        sal = center.find('div', {'column': '10'})
        for s in sal.find_all('div', {'class': 'webix_cell'}):

            dk_sal = s.text

            # trim dollar sign
            dk_sal = dk_sal[1:]
            salaries.append(dk_sal)

        # add salary to the current dictionary
        for i in range(len(names)):

            if names[i] in self.name_dict:

                self.name_dict[names[i]].append(salaries[i])

    def get_all_current_data(self):

        self.scrape_current_fd_data()
        self.scrape_current_dk_salary()

        return self.name_dict

    def get_all_historical_data(self):
        """
        :return: dict: {player: [position, team, opposition, fanduel pts, fd salary, dk salary]}
        """

        self.scrape_historical_fd_data()
        self.scrape_historical_dk_salary()

        return self.name_dict

########################################################################################################################

class PlayerDisplay:

    def __init__(self, master, player_object):

        self.master = master

        # initialize player obj
        self.player = player_object

        # load dataframes
        self.player_df, self.pos_df = self.get_dataframes()

        # widget framework
        self.frame = tk.Frame(self.master)
        self.player_stats_window = tk.Toplevel(self.frame)
        self.player_stats_window.title(f'{self.player.name}' + ": " + f'{self.player.position}')
        self.player_stats_window.geometry('600x400')
        self.tab_main = ttk.Notebook(self.player_stats_window)

        # display tab widgets
        self.display_stats_overview_tab()
        self.display_salary_tab()
        self.display_point_tab()

    def get_dataframes(self):
        """

        :return: player-specific dataframe and a position-specific dataframe
        """
        conn = db.connect('data.db')
        curs = conn.cursor()
        df = pd.read_sql_query("SELECT * FROM Games", conn)

        player_df = df[(df['player'] == self.player.name) &
                       (df['season'] == int(self.player.season)) &
                       (df['week'] <= int(self.player.week))]

        # dataframe for the position for the entire season
        pos_df = df[(df['position'] == self.player.position) &
                    (df['season'] == int(self.player.season)) &
                    (df['week'] <= int(self.player.week))]

        conn.close()

        return player_df, pos_df

    def display_stats_overview_tab(self):

        stats_tab = ttk.Frame(self.tab_main)
        self.tab_main.add(stats_tab, text="Overview")
        self.tab_main.pack(expand=1, fill="both")

        running_avg = np.round(self.player_df['fd_points'].mean(), 3)
        pos_avg = np.round(self.pos_df['fd_points'].mean(), 3)
        psal_avg = np.round(self.pos_df['fdsal'].mean(), 0)
        # TODO division by zero error with some players traced to rel_value
        rel_value = (int(self.player.fd_sal) / (int(running_avg))*100)
        formatted_nn = np.round(self.player.proj_pts, 2)
        games = self.player_df['week']
        games_percentage = np.round(len(games)/int(self.player.week), 3)

        # TODO format this and make it look nicer
        overview = tk.Text(stats_tab)
        ov_text = "Player Stats\n"
        line_text = "------------------\n"
        name_text = "Name: " + f"{self.player.name}\n"
        team_text = "Team: " + f"{self.player.team}\n"
        pos_text = "Position: " + f"{self.player.position}\n"
        sal_text = "Salary: " + f"{self.player.fd_sal}\n"
        svg_text = "Position Sal Avg: " + f"{psal_avg}\n"
        avg_text = "Average Pts: " + f"{running_avg}\n"
        pd_text = "Predicted Pts: " + f"{formatted_nn}\n"
        pavg_text = "Position Pt Avg: " + f"{pos_avg}\n"
        rval_text = "Dollars per pt: " + f"{rel_value}\n"
        gplay_text = "Play percentage: " + f"{games_percentage}\n"

        text_list = [ov_text, line_text, name_text, team_text,
                     pos_text, sal_text, svg_text, avg_text,
                     pd_text, pavg_text, rval_text, gplay_text]
        for i in text_list:
            overview.insert(tk.INSERT, i)

        # disable to make read-only otherwise user can change it
        overview.config(state='disabled')
        overview.pack()

    def display_salary_tab(self):

        player_salaries = list(self.player_df['fdsal'])
        weeks_played = list(self.player_df['week'])
        sal_cap = self.pos_df['fdsal'].max()
        sal_min = self.pos_df['fdsal'].min()
        salary_tab = ttk.Frame(self.tab_main)
        self.tab_main.add(salary_tab, text="Salary")
        self.tab_main.pack(expand=1, fill="both")

        total = [i for i in range(17)]
        x = np.arange(1, len(total)+1)
        y = np.arange(sal_min, sal_cap, 500)
        salary_figure = plt.Figure(figsize=(6, 5), dpi=100)
        ax1 = salary_figure.add_subplot(111)
        ax1.bar(weeks_played, player_salaries, width=.4)
        ax1.set_xticks(x)
        ax1.set_yticks(y)

        canvas = FigureCanvasTkAgg(salary_figure, salary_tab)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        ax1.set_title('Salary per week')
        ax1.set_xlim(0, 18)
        ax1.set_ylim(sal_min, sal_cap)

    def display_point_tab(self):

        points_tab = ttk.Frame(self.tab_main)
        self.tab_main.add(points_tab, text="Points")
        self.tab_main.pack(expand=1, fill="both")

        raw_nn_scores = list(self.player_df['nn_score'])
        raw_fd_points = list(self.player_df['fd_points'])
        weeks_played = list(self.player_df['week'])

        # data for setting y range
        pt_max = self.pos_df['fd_points'].max()

        total = [i for i in range(1, 18)]
        fd_points = []
        nn_points = []
        # create scores lists with 0s for all games not played
        for n in total:
            if n in weeks_played:
                i = weeks_played.index(n)
                fd_points.append(raw_fd_points[i])
                nn_points.append(raw_nn_scores[i])
            else:
                fd_points.append(0)
                nn_points.append(0)

        x = np.arange(1, len(total)+1)
        y = np.arange(-2, pt_max, 2)
        point_figure = plt.Figure(figsize=(6, 5), dpi=100)
        ax2 = point_figure.add_subplot(111)
        ax2.bar(x, fd_points, color='blue', width=.3, align='center', label='Actual Scored')
        ax2.bar(x + .3, nn_points, color='red', width=.3, label='Predicted')
        ax2.legend(loc='upper right')
        canvas2 = FigureCanvasTkAgg(point_figure, points_tab)
        canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        ax2.set_xticks(x)
        ax2.set_yticks(y)
        ax2.set_title('Predicted vs Actual')
        ax2.set_xlim(0, 18)
        ax2.set_ylim(0, pt_max)

########################################################################################################################


class Lineup:

    def __init__(self, test_year, test_week):

        self.datafetcher = DataFetcher()
        self.test_year = test_year
        self.test_week = test_week
        self.total_pts = 0
        self.total_salary = 0

        # vars needed for optimizer
        self.players = []
        self.positions = []
        self.salaries = []
        self.nn_scores = []
        self.fd_pts = []

        self.lineup_official = []
        self.maximized_scores = []
        self.lineup_salaries = []
        self.lineup_positions = []

        # lineup constraints: upper and lower limits for each position
        self.lineup_lim = 9
        self.sal_lim = 60000
        self.qb_lim = 1
        self.rb_lower_lim = 2
        self.rb_upper_lim = 3
        self.wr_lower_lim = 3
        self.wr_upper_lim = 4
        self.te_lower_lim = 1
        self.te_upper_lim = 2
        self.def_lim = 1

        # load data into individual lists
        self.update_pool()

    def update_pool(self):

        # reset variables
        self.players = []
        self.positions = []
        self.salaries = []
        self.nn_scores = []
        self.fd_pts = []

        query = """
                    SELECT player, position, fdsal, nn_score, fd_points
                    FROM Games 
                    WHERE season = ?
                    AND week = ?
                    AND will_play = 1"""

        result = self.datafetcher.fetch_query(query, [self.test_year, self.test_week])

        # lists used for assessing optimal lineup
        for i in result:

            self.players.append(i[0])
            self.positions.append(i[1])
            self.salaries.append(i[2])
            self.nn_scores.append(i[3])
            self.fd_pts.append(i[4])

    def get_total_points(self):

        self.total_pts = 0
        for p in self.lineup_official:

            self.total_pts += p.proj_pts

        return self.total_pts

    def get_total_salary(self):

        self.total_salary = 0
        for p in self.lineup_official:

            self.total_salary += int(p.fd_sal)

        return self.total_salary

    def remove_from_pool(self, player_object):

        # player index the same for all lists
        index = self.players.index(player_object.name)

        del self.players[index]
        del self.nn_scores[index]
        del self.positions[index]
        del self.salaries[index]

        self.lineup_positions.append(player_object.position)

    def add_to_pool(self, player_object):

        self.players.append(player_object.name)
        self.nn_scores.append(float(player_object.proj_pts))
        self.positions.append(player_object.position)
        self.salaries.append(int(player_object.fd_sal))

        self.lineup_positions.remove(player_object.position)

    def test_parameters(self):
        print("qb_lim: ", self.qb_lim)
        print("rb: ", self.rb_upper_lim, self.rb_lower_lim)
        print("wr: ", self.wr_upper_lim, self.wr_lower_lim)
        print("te: ", self.te_upper_lim, self.te_lower_lim)
        print("def: ", self.def_lim)
        print("sal: ", self.sal_lim)
        print("lineup lim: ", self.lineup_lim)
        print(len(self.lineup_official))

    def print_lineup(self):

        for player in self.lineup_official:

            print(player.name)

    def get_lineup(self):

        return self.lineup_official

    def passes_constraints(self, player_object):

        if player_object.name not in self.players:

            return False

        # see if the player's been added already
        for player in self.lineup_official:

            if player.name == player_object.name:

                return False

        # if lineup is already full
        if len(self.lineup_official) >= 9:

            return False

        # these positions are eligible for flex slot so we need to know their count
        rb_count = self.lineup_positions.count('RB')
        wr_count = self.lineup_positions.count('WR')
        te_count = self.lineup_positions.count('TE')

        if player_object.position == 'QB':

            if self.qb_lim == 0:

                return False

            else:

                # meets constraints
                if int(player_object.fd_sal) <= self.sal_lim:

                    return True

                else:

                    return False

        if player_object.position == 'RB':

            if rb_count == 3:

                return False

            elif rb_count == 2:

                # if flex slot already filled by wr or te
                if wr_count > 3 or te_count > 1:

                    return False

                else:

                    if int(player_object.fd_sal) <= self.sal_lim:

                        return True

                    else:

                        return False

            elif rb_count < 2:

                # meets constraints
                if int(player_object.fd_sal) <= self.sal_lim:

                    return True

                else:

                    return False

        if player_object.position == 'WR':

            if wr_count == 4:

                return False

            elif wr_count == 3:

                # if flex slot already filled by rb or te
                if rb_count > 3 or te_count > 1:

                    return False

                else:

                    if int(player_object.fd_sal) <= self.sal_lim:

                        return True

                    else:

                        return False

            elif wr_count < 3:

                if int(player_object.fd_sal) <= self.sal_lim:

                    return True

                else:

                    return False

        if player_object.position == 'TE':

            if te_count == 2:

                return False

            elif te_count == 1:

                # if flex slot already filled by wr or rb
                if wr_count > 3 or rb_count > 2:

                    return False

                else:

                    # meets constraints
                    if int(player_object.fd_sal) <= self.sal_lim:

                        return True

                    else:

                        return False

            elif te_count < 1:

                # meets constraints
                if int(player_object.fd_sal) <= self.sal_lim:

                    return True

                else:

                    return False

        if player_object.position == 'DEF':

            if self.def_lim == 0:

                return False

            else:

                # meets constraints
                if int(player_object.fd_sal) <= self.sal_lim:

                    return True

                else:

                    return False

    def adjust_constraints_selecting(self, player_object):

        # determine position and subtract position constraint
        if player_object.position == 'QB':

            self.qb_lim -= 1

        elif player_object.position == 'RB':

            if self.rb_upper_lim == 1:

                self.rb_upper_lim -= 1

            else:

                self.rb_upper_lim -= 1
                self.rb_lower_lim -= 1

        elif player_object.position == 'WR':

            if self.wr_upper_lim == 1:

                self.wr_upper_lim -= 1

            else:

                self.wr_upper_lim -= 1
                self.wr_lower_lim -= 1

        elif player_object.position == 'TE':

            if self.te_upper_lim == 1:

                self.te_upper_lim -= 1

            else:

                self.te_upper_lim -= 1
                self.te_lower_lim -= 1

        elif player_object.position == 'DEF':

            self.def_lim -= 1

        # adjust salary constraint
        self.sal_lim -= int(player_object.fd_sal)

        # adjust lineup limit
        self.lineup_lim -= 1

    def adjust_constraints_removing(self, player_object):

        # determine player position and add back position limit
        if player_object.position == 'QB':

            self.qb_lim += 1

        elif player_object.position == 'RB':

            if self.rb_upper_lim == 0:

                self.rb_upper_lim += 1

            else:

                self.rb_upper_lim += 1
                self.rb_lower_lim += 1

        elif player_object.position == 'WR':

            if self.wr_upper_lim == 0:

                self.wr_upper_lim += 1

            else:

                self.wr_upper_lim += 1
                self.wr_lower_lim += 1

        elif player_object.position == 'TE':

            if self.te_upper_lim == 0:

                self.te_upper_lim += 1

            else:

                self.te_upper_lim += 1
                self.te_lower_lim += 1

        elif player_object.position == 'DEF':

            self.def_lim += 1

        # adjust salary constraint
        self.sal_lim += int(player_object.fd_sal)

        # adjust lineup limit
        self.lineup_lim += 1

    def generate_lineup(self):

        # temporarily stores lineup before officially selecting it
        temp_lineup = []

        # initialize problem
        prob = pulp.LpProblem("Lineup", pulp.LpMaximize)

        # create dictionaries or list of indices
        player_index = [i for i in range(len(self.players))]
        max_scores = {i: self.nn_scores[i] for i in range(len(self.nn_scores))}
        cost = {i: self.salaries[i] for i in range(len(self.salaries))}
        qb = {i: 1 if self.positions[i] == 'QB' else 0 for i in range(len(self.players))}
        rb = {i: 1 if self.positions[i] == 'RB' else 0 for i in range(len(self.players))}
        wr = {i: 1 if self.positions[i] == 'WR' else 0 for i in range(len(self.players))}
        te = {i: 1 if self.positions[i] == 'TE' else 0 for i in range(len(self.players))}
        df = {i: 1 if self.positions[i] == 'DEF' else 0 for i in range(len(self.players))}

        decision_var = pulp.LpVariable.dict("Players", player_index, 0, 1, cat="Binary")

        # score to be maximized
        prob += pulp.lpSum([max_scores[i] * decision_var[i] for i in player_index]), "Total Score"

        # subject to these constraints
        prob += pulp.lpSum([decision_var[i] for i in player_index]) == self.lineup_lim
        prob += pulp.lpSum([cost[i] * decision_var[i] for i in player_index]) <= self.sal_lim
        prob += pulp.lpSum([qb[i] * decision_var[i] for i in player_index]) == self.qb_lim
        prob += pulp.lpSum([rb[i] * decision_var[i] for i in player_index]) >= self.rb_lower_lim
        prob += pulp.lpSum([rb[i] * decision_var[i] for i in player_index]) <= self.rb_upper_lim
        prob += pulp.lpSum([wr[i] * decision_var[i] for i in player_index]) >= self.wr_lower_lim
        prob += pulp.lpSum([wr[i] * decision_var[i] for i in player_index]) <= self.wr_upper_lim
        prob += pulp.lpSum([te[i] * decision_var[i] for i in player_index]) >= self.te_lower_lim
        prob += pulp.lpSum([te[i] * decision_var[i] for i in player_index]) <= self.te_upper_lim
        prob += pulp.lpSum([df[i] * decision_var[i] for i in player_index]) == self.def_lim

        try:

            prob.solve()

            for i in range(len(self.players)):

                # if the player is chosen
                if decision_var[i].varValue:

                    # create player object and put in temp list
                    player_object = Player(self.players[i], self.test_year, self.test_week)
                    player_object.set_player_data()
                    temp_lineup.append(player_object)

            # insert temp list into official lineup list
            for p in temp_lineup:

                self.select_player(p)

        except pulp.PulpSolverError:

            # TODO don't just put a print statement. Add an actual exception
            print("infeasible")

    def select_player(self, player_object):

        if self.passes_constraints(player_object):

            # add player obj to lineup
            self.lineup_official.append(player_object)

            # delete these values from lists so it doesn't mess with lineup optimizer
            self.remove_from_pool(player_object)

            # adjust position, lineup, and salary limits
            self.adjust_constraints_selecting(player_object)

            # returns True so lineup tree knows it can insert this player
            return True

        else:

            return False

    def remove_player(self, player_object):

        # find player and delete from lineup
        for i, player in enumerate(self.lineup_official):

            # if the player is equal to player object
            if player.__eq__(player_object):

                # place data back into lists for future use by optimizer
                self.add_to_pool(player)

                # remove from lineup
                del self.lineup_official[i]

                # adjust constraints
                self.adjust_constraints_removing(player)
