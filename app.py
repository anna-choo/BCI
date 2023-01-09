import pandas as pd
import os, collections, csv, random, sys
import numpy as np
from sympy import Point3D, Line3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.pyplot import figure
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtChart import *
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import scipy.stats as st
from tqdm import tqdm
import itertools
from numpy import exp
from statistics import mean
import scipy.stats as stats
import pickle
from typing import OrderedDict






class BCI_Interface(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setFixedSize(900, 500)
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.setWindowTitle('BCI')

        self.LABEL_normative = QLabel(self.widget)
        self.LABEL_normative.setText("NORMATIVE")
        self.LABEL_normative.setGeometry(QRect(40,30,150,20))
        # self.LABEL_normative.setFont(QSize)

        self.LABEL_file1 = QLabel(self.widget)
        self.LABEL_file1.setText("Towel Folding Down: ")
        self.LABEL_file1.setGeometry(QRect(40,70,120,20))

        self.file1_path = QLineEdit(" Please Select", self.widget)
        self.file1_path.setStyleSheet("background-color: White;")
        self.file1_path.setGeometry(QRect(160, 65, 600, 30))

        self.BUTTON_browse_file1 = QPushButton("Browse", self.widget)
        self.BUTTON_browse_file1.setGeometry(QRect(780, 65, 80, 30))
        # self.BUTTON_browse_file1.clicked.connect(self.browse_files1)


        self.LABEL_file2 = QLabel(self.widget)
        self.LABEL_file2.setText("Towel Folding Across: ")
        self.LABEL_file2.setGeometry(QRect(40, 120 , 120, 20))

        self.file2_path = QLineEdit(" Please Select", self.widget)
        self.file2_path.setStyleSheet("background-color: White;")
        self.file2_path.setGeometry(QRect(160, 115, 600, 30))

        self.BUTTON_browse_file2 = QPushButton("Browse", self.widget)
        self.BUTTON_browse_file2.setGeometry(QRect(780, 115, 80, 30))
        # self.BUTTON_browse_file2.clicked.connect(self.browse_files2)



        
        self.LABEL_file3 = QLabel(self.widget)
        self.LABEL_file3.setText("Grasp: ")
        self.LABEL_file3.setGeometry(QRect(40, 170, 120, 20))

        self.file3_path = QLineEdit(" Please Select", self.widget)
        self.file3_path.setStyleSheet("background-color: White;")
        self.file3_path.setGeometry(QRect(160, 165, 600, 30))

        self.BUTTON_browse_file3 = QPushButton("Browse", self.widget)
        self.BUTTON_browse_file3.setGeometry(QRect(780, 165, 80, 30))
        # self.BUTTON_browse_file3.clicked.connect(self.browse_files1)





        self.LABEL_file4 = QLabel(self.widget)
        self.LABEL_file4.setText("Lateral: ")
        self.LABEL_file4.setGeometry(QRect(40, 220, 120, 20))

        self.file4_path = QLineEdit(" Please Select", self.widget)
        self.file4_path.setStyleSheet("background-color: White;")
        self.file4_path.setGeometry(QRect(160, 215, 600, 30))

        self.BUTTON_browse_file4 = QPushButton("Browse", self.widget)
        self.BUTTON_browse_file4.setGeometry(QRect(780, 215, 80, 30))
        # self.BUTTON_browse_file4.clicked.connect(self.browse_files1)






        # self.generateGraphButton = QPushButton("Graph", self.widget)
        # self.generateGraphButton.setGeometry(QRect(420, 120, 90, 30))
        # self.generateGraphButton.clicked.connect(self.plot_graph)







        ROM_dict = collections.defaultdict()
        ROM_dict[('elbow', 'X')] = tuple((0,140))
        ROM_dict['elbow', 'Z'] = tuple((-80, 80))
        ROM_dict[('shoulder', 'X')] = tuple((-50, 180))
        ROM_dict[('shoulder', 'Y')] = tuple((-180, 50))
        ROM_dict[('shoulder', 'Z')]= tuple((-90, 90))
        ROM_dict[('wrist', 'X')] = tuple((-60, 60))
        # ROM_dict[('wrist', 'Z')] = tuple((-20, 20))
        data_point = 300

        # inputs
        TOWEL_vertFold = {"shoulder":["X"], "elbow":["X"], "wrist":["X"]}
        TOWEL_horiFold = {"shoulder":["X", "Y"], "elbow":["X"], "wrist":["X"]}
        GRASP = {"shoulder":["X"], "elbow":["X"], "wrist":["X"]}
        LATERAL = {"shoulder":["X", "Y"], "elbow":["X"]}
        MOUTH = {"shoulder":["X", "Y"], "elbow":["X", "Z"], "wrist":["X"]} 
        KEY = {"shoulder":["X", "Y"], "elbow":["X", "Z"]}

        # analyse_dict_L = {"shoulder": ["L_Shoulder_JOINT_ANGLE"], "elbow": ["L_Elbow_JOINT_ANGLE"], "wrist": ["L_Wrist_JOINT_ANGLE"]}
        # analyse_dict_R = {"shoulder": ["R_Shoulder_JOINT_ANGLE"], "elbow": ["R_Elbow_JOINT_ANGLE"], "wrist": ["R_Wrist_JOINT_ANGLE"]}


        # normative_towel_vert_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_towel\RightJA_2Pick1_Bwd.txt", sep = '\t')
        # stroke_towel_vert_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_towel_vert_R", sep = '\t')

        # normative_towel_hori_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_towel\RightJA_4Pick2_Cross.txt", sep = '\t')
        # stroke_towel_hori_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_towel_hori_R", sep = '\t')

        # normative_grasp_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_grasp\RightJA_2Block_Shelf.txt", sep = '\t')
        # stroke_grasp_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_grasp_R", sep = '\t')

        # normative_lateral_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_lateral\RightJA_1Start_Cross.txt", sep = '\t')
        # stroke_lateral_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_lateral_R", sep = '\t')

        # normative_mouth_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_mouth\RightJA_1Start_Mouth.txt", sep = '\t') # input
        # stroke_mouth_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_mouth_R", sep = '\t')


        # normative_key_R = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\BCI001\Normative\1. EXPORTS\EXPORT_keystand\RightJA_2Touchkey_Turn.txt", sep = '\t')
        # stroke_key_R = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_key_R", sep = '\t')



        # towelvert_R_df, result_towel_vert_R, norm_dist_towel_vert_R = self.main(normative_towel_vert_R, stroke_towel_vert_R, TOWEL_vertFold, analyse_dict_R, ROM_dict, data_point, 'towel vert R')

        # towelhori_R_df, result_towel_hori_R, norm_dist_towel_hori_R = self.main(normative_towel_hori_R, stroke_towel_hori_R, TOWEL_horiFold, analyse_dict_R, ROM_dict, data_point, 'towel hori R')

        # grasp_R_df, result_grasp_R, norm_dist_grasp_R = self.main(normative_grasp_R, stroke_grasp_R, GRASP, analyse_dict_R, ROM_dict, data_point, 'grasp R')

        # lateral_R_df, result_lateral_R, norm_dist_lateral_R = self.main(normative_lateral_R, stroke_lateral_R, LATERAL, analyse_dict_R, ROM_dict, data_point, 'lateral R')

        # mouth_R_df, result_mouth_R, norm_dist_mouth_R = self.main(normative_mouth_R, stroke_mouth_R, MOUTH, analyse_dict_R, ROM_dict, data_point, 'mouth R')

        # key_R_df, result_key_R, norm_dist_key_R = self.main(normative_key_R, stroke_key_R, KEY, analyse_dict_R, ROM_dict, data_point, 'key R')





    def manipulate_data(self, raw, marker):
        '''anchor data header to the marker given'''
        df = raw
        df.drop([1,2], inplace = True)
        subject = [i[0:5] for i in df.columns]
        subject[0] = "subject"
        df.loc[-1] = subject
        df = df.sort_index()
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace = True)
        df.rename(columns = {df.columns[0] : "video"}, inplace = True)
        df["video"][1] = "markers"
        df["video"][2] = "axis"
        df_transposed = df.T
        df_transposed.columns = df_transposed.iloc[0]
        df_transposed.drop('video', axis = 0, inplace = True)
        marker_list = df_transposed["markers"].unique()
        subject_list = df_transposed["subject"].unique()
        DICT_repetition = collections.defaultdict()
        for subject in subject_list:
            repetition = int(sum((df_transposed["markers"] == marker) & (df_transposed["subject"] == subject)) / 3)
            DICT_repetition[subject] = repetition
        
        standard_marker_list = ['C7', 'LACR', 'LASIS', 'LCAP', 'LFA1', 'LFA2', 'LFA3', 'LHEAD',
            'LHLE', 'LHMC1', 'LHMC2', 'LHMC3', 'LHME', 'LICR', 'LPSIS', 'LRSP',
            'LTEMP', 'LUA1', 'LUA2', 'LUA3', 'LUA4', 'LUSP', 'RACR', 'RASIS',
            'RCAP', 'RFA1', 'RFA2', 'RFA3', 'RHEAD', 'RHLE', 'RHMC1', 'RHMC2',
            'RHMC3', 'RHME', 'RICR', 'RPSIS', 'RRSP', 'RTEMP', 'RUA1', 'RUA2',
            'RUA3', 'RUA4', 'RUSP', 'STER', 'T10', 'T4', 'T7', 'XPRO',
            'LMIDDLE', 'LTHUMB', 'RMIDDLE', 'RTHUMB',
            'Pelvis_JOINT_ACCELERATION', 'Pelvis_JOINT_ANGLE', 'T8',
            'L_Elbow_JOINT_ACCELERATION', 'L_Elbow_JOINT_ANGLE',
            'L_Shoulder_JOINT_ACCELERATION', 'L_Shoulder_JOINT_ANGLE',
            'L_Wrist_JOINT_ACCELERATION', 'L_Wrist_JOINT_ANGLE',
            'Neck_JOINT_ACCELERATION', 'Neck_JOINT_ANGLE',
            'R_Elbow_JOINT_ACCELERATION', 'R_Elbow_JOINT_ANGLE',
            'R_Shoulder_JOINT_ACCELERATION', 'R_Shoulder_JOINT_ANGLE',
            'R_Wrist_JOINT_ACCELERATION', 'R_Wrist_JOINT_ANGLE',
            'Waist_JOINT_ACCELERATION', 'Waist_JOINT_ANGLE']
        discard_index_list = []
        for i in marker_list:
            if i not in standard_marker_list:
                discard_index_list.extend(df_transposed[df_transposed['markers'] == i].index)
        df_transposed.drop(discard_index_list, axis = 0, inplace = True)
        
        subject_marker_dict = collections.defaultdict()
        repetition_list = []
        for subject in subject_list:
            subject_marker_list = df_transposed[df_transposed["subject"] == subject]["markers"].unique()
            subject_marker_dict[subject] = subject_marker_list
            for i in range(len(subject_marker_list)):
                for j in range(DICT_repetition[subject]):
                    repetition_list.extend([j + 1, j + 1, j + 1])
        df_transposed.insert(3, 'repetition', repetition_list)



        return df_transposed, subject_list, subject_marker_dict, DICT_repetition


    def get_coordinates_dict(self, data, data_point, point_to_analyse, subject_list, repetition_dict, subject_marker_dict):
        ''' 
        first output: separate out the coordinates for each repetition for each subject;
        second output: calculate out the mean trajectory;
        third output: total number of repetitions considered
        (not yet in use)
        '''
        # separate out the coordinates for each repetition for each subject
        subject_coordinates_repetition_dict = collections.defaultdict()
        coordinates_array = None
        n_sample = 0
        for subject in subject_list:
            # only if the subject has the point to analyse, otherwise, skipped
            if point_to_analyse in subject_marker_dict[subject]:
                n_sample = n_sample + repetition_dict[subject]

                for i in range(repetition_dict[subject]):
                    x_data, y_data = np.array(data[(data["subject"] == subject) & (data["markers"] == point_to_analyse) & (data["axis"] == "X") & (data["repetition"] == i + 1)].drop(columns = ["subject", "markers", "repetition", "axis"]).T).astype(float), np.array(data[(data["subject"] == subject) & (data["markers"] == point_to_analyse) & (data["axis"] == "Y") & (data["repetition"] == i + 1)].drop(columns = ["subject", "markers", "repetition", "axis"]).T).astype(float)
                    a, b = x_data[0], y_data[0]
                    x_data = x_data - a
                    y_data = y_data - b
                    coordinates = np.concatenate((x_data, y_data), axis = 1)
                    if coordinates_array is None:
                        coordinates_array = coordinates
                    elif  coordinates_array is not None:
                        coordinates_array = np.append(coordinates_array, coordinates)
                
                # compile into each repetition for each subject
                subject_coordinates_repetition_dict[subject] = coordinates_array.reshape(-1, data_point,2)
        
        # calculate the trajectory mean
        coordinates_mean = np.mean(np.concatenate(list(subject_coordinates_repetition_dict.values())), axis = 0)

        return subject_coordinates_repetition_dict, coordinates_mean, n_sample


    def get_joint_angle_dict(self, data, data_point, JA_to_analyse, axis_to_analyse, subject_list, repetition_dict, subject_marker_dict):
        ''' 
        first output: separate out the JA for each repetition for each subject;
        second output: calculate out the mean JA trajectory;
        third output: total number of repetitions considered
        '''
        # separate out the JA for each repetition for each subject
        subject_JA_repetition_dict = collections.defaultdict()
        JA_array = None
        n_sample = 0
        for subject in subject_list:
            if JA_to_analyse in subject_marker_dict[subject]:
                n_sample = n_sample + repetition_dict[subject]

                for i in range(repetition_dict[subject]):
                    JA = np.array(data[(data["subject"] == subject) & (data["markers"] == JA_to_analyse) & (data["axis"] == axis_to_analyse) & (data["repetition"] == i + 1)].drop(columns = ["subject", "markers", "repetition", "axis"]).T).astype(float)
                    # for x-coordinates(time)
                    time = np.array([ int(i) for i in range(JA.shape[0])]).reshape(JA.shape[0],-1)
                    time_series_coordinates = np.concatenate((time, JA), axis = 1)
                    if JA_array is None:
                        JA_array = time_series_coordinates
                    elif  JA_array is not None:
                        JA_array = np.append(JA_array, time_series_coordinates)
                    
                # compile into each repetition for each subject
                subject_JA_repetition_dict[subject] = JA_array.reshape(-1, data_point, 2)

        # calculate the trajectory mean
        JA_mean = np.mean(np.concatenate(list(subject_JA_repetition_dict.values())), axis = 0)

        return subject_JA_repetition_dict, JA_mean, n_sample




    def generate_norm_dist(self, normal_dist_data, show_graph = False):
        ''' to generate normal distribution for each point to analyse and return the confidence interval of 95%, show graph if needed'''
        confidence = st.t.interval(alpha=0.95, df=len(normal_dist_data)-1, loc=np.mean(normal_dist_data), scale=st.sem(normal_dist_data))
        if show_graph:
            count, bins, ignored = plt.hist(list(normal_dist_data), len(normal_dist_data))
            plt.show()
        return confidence

    def generate_score(normative_mean, stroke_mean, plot_graph = False):
        '''calculate dtw score between two mean graph, plot graph if needed to see the time warping relation'''
        score = dtw.distance(normative_mean[:,1], stroke_mean[:,1])
        if plot_graph:
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
            path = dtw.warping_path(normative_mean[:,1], stroke_mean[:,1])
            dtwvis.plot_warping(normative_mean[:,1], stroke_mean[:,1], path, fig = fig, axs = axs)
            plt.show()
        return score



    def get_feature_dict(self, analyse_dict, axis_list):
        feature_dict = collections.defaultdict()
        for key in analyse_dict.keys():
            try:
                for i in analyse_dict[key]:
                    feature_dict[i] = axis_list[key]
            except:
                continue
        return feature_dict




    def upper_threshold(self, result_df):
        result_list = []
        for index in range(len(result_df)):
            if (int(result_df["Stroke Patient Score"][index]) > result_df["Normative Confidence Interval"][index][0]) == True and (int(result_df["Stroke Patient Score"][index]) < result_df["Normative Confidence Interval"][index][1]) == True:
                # result_df["Drop (Upper threshold)"][index] = 1
                result_list.append(1)
            else:
                # result_df["Drop (Upper threshold)"][index] = 0
                result_list.append(0)

        result_df['Drop (Upper threshold)'] = result_list
        return result_df

    def get_trial_ROM(self, df, feature_dict, first_result):
        df['feature'] = np.array(df["markers"] + '_'+ df["axis"])
        features_list = [i+'_'+j for i, k in feature_dict.items() for j in k]
        relevant_df = df[df["feature"].isin(features_list)]
        trial_rom_list = []
        for i in range(len(relevant_df)):
            trial_rom_list.append(max([float(k) for k in relevant_df.iloc[i,4:-1]]) - min([float(k) for k in relevant_df.iloc[i,4:-1]]))
            # print(trial_ROM)
        # test.iloc[:,4:-1]
        relevant_df["trial ROM"] = trial_rom_list
        average_list = collections.defaultdict()
        for feature in features_list:
            average_list[feature] = mean(relevant_df["trial ROM"][relevant_df["feature"] == feature])
        resorted_average_list = []
        for i in range(len(first_result)):
            if first_result["Feature"][i] == "Flexion/Extension":
                axis = "X"
            if first_result["Feature"][i] == "Abduction/Adduction":
                axis = "Y"
            if first_result["Feature"][i] == "External/Internal Rotation":
                axis = "Z"
            resorted_average_list.append(average_list[first_result["Joint Angle"][i] + '_' + axis])
        first_result["Trial ROM"] = resorted_average_list
        return relevant_df, features_list, first_result
        return df

    def get_significance_vector(self, result_table, feature_dict, analyse_dict, ROM_dict):
        significancy_list = []
        rom_list = []
        total_calculation = []
        result_table = result_table.reset_index()
        for index in range(len(result_table)):
            for each_axis in feature_dict[result_table["Joint Angle"][index]]:
                for i, k in analyse_dict.items():
                    if result_table["Joint Angle"][index] in k:
                        rangeOfMotion = tuple((i, each_axis))
                        break
            rom_list.append(ROM_dict[rangeOfMotion])
            if int(result_table["Drop (Upper threshold)"][index]) == 0:
                significancy = (result_table["Trial ROM"][index]) / (int(ROM_dict[rangeOfMotion][1]) - int(ROM_dict[rangeOfMotion][0]))
                significancy_list.append(float(significancy))
            else:
                significancy_list.append(0)
        # print(significancy_list)
        result_table["ROM"] = rom_list
        result_table["weight"] = significancy_list

        return result_table


app = QApplication(sys.argv)

window = BCI_Interface()


window.show()

app.exec()