import pandas as pd
import collections
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
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
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import csv 


class BCI_analysis:
    def __init__(self, affected_side):
        self.ROM_dict = collections.defaultdict()
        self.ROM_dict[('elbow', 'X')] = tuple((0,140))
        self.ROM_dict['elbow', 'Z'] = tuple((-80, 80))
        self.ROM_dict[('shoulder', 'X')] = tuple((-50, 180))
        self.ROM_dict[('shoulder', 'Y')] = tuple((-180, 50))
        self.ROM_dict[('shoulder', 'Z')]= tuple((-90, 90))
        self.ROM_dict[('wrist', 'X')] = tuple((-60, 60))
        #('wrist', 'Z')] = tuple((-20, 20))
        self.data_point = 300 # number of time frame normalized to

        # inputs
        self.TOWEL_vertFold = {"shoulder":["X"], "elbow":["X"], "wrist":["X"]}
        self.TOWEL_horiFold = {"shoulder":["X", "Y"], "elbow":["X"], "wrist":["X"]}
        self.GRASP = {"shoulder":["X"], "elbow":["X"], "wrist":["X"]}
        self.LATERAL = {"shoulder":["X", "Y"], "elbow":["X"]}
        self.MOUTH = {"shoulder":["X", "Y"], "elbow":["X", "Z"], "wrist":["X"]} 
        self.KEY = {"shoulder":["X", "Y"], "elbow":["X", "Z"]}

        self.task_dict = {
            "towel_JA_2pick1_bwd":self.TOWEL_vertFold,
            "towel_JA_4Pick2_cross":self.TOWEL_horiFold,
            "grasp_JA_2block_shelf":self.GRASP,
            "lateral_JA_1start_cross":self.LATERAL,
            "mouth_JA_1start_mouth":self.MOUTH,
            "key_JA_3turn_return":self.KEY,
            "key_JA_2touchkey_turn":self.KEY
        }

        self.NHG_task_mapping = {
            "towel_JA_2pick1_bwd":"Moving Towel",
            "towel_JA_4Pick2_cross":"Picking Block",
            "grasp_JA_2block_shelf":"Shelving Items",
            "lateral_JA_1start_cross":"Scanning Goods",
            "mouth_JA_1start_mouth":"Eating",
            "key_JA_3turn_return":"Pouring Drinks Part1",
            "key_JA_2touchkey_turn":"Pouring Drinks Part2"
        }

        if affected_side == "Left":
            self.analyse_dict = {"shoulder": ["L_Shoulder_JOINT_ANGLE"], "elbow": ["L_Elbow_JOINT_ANGLE"], "wrist": ["L_Wrist_JOINT_ANGLE"]}
        elif affected_side == "Right":
            self.analyse_dict = {"shoulder": ["R_Shoulder_JOINT_ANGLE"], "elbow": ["R_Elbow_JOINT_ANGLE"], "wrist": ["R_Wrist_JOINT_ANGLE"]}


    def manipulate_data(self, raw, stroke = False):
        '''anchor data header to the marker given'''
        df = raw
        # print(df)
        df.drop([1,2], inplace = True)
        if stroke == True:
            subject = ["subject"]
            for i in df.columns[1:]:
                subject.append(i.split("\\")[1]) 

        else:
            subject = [i[0:5] for i in df.columns]
            subject[0] = "subject"
        df.loc[-1] = subject
        videoRecord = []
        for i in df.columns:
            videoRecord.append(i.split("\\")[-1].split(".")[0])
        videoRecord[0] = "record"
        df.loc[-2] = videoRecord
        df = df.sort_index()
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace = True)
        df.rename(columns = {df.columns[0] : "video"}, inplace = True)
        df["video"][2] = "markers"
        df["video"][3] = "axis"
        df_transposed = df.T
        df_transposed.columns = df_transposed.iloc[0]
        df_transposed.drop('video', axis = 0, inplace = True)
        marker_list = df_transposed["markers"].unique()
        subject_list = df_transposed["subject"].unique()
        # print(df_transposed)
        # print(subject_list)
        standard_marker_list = [
            'L_Elbow_JOINT_ANGLE',
            'L_Shoulder_JOINT_ANGLE',
            'L_Wrist_JOINT_ANGLE',
            'R_Elbow_JOINT_ANGLE',
            'R_Shoulder_JOINT_ANGLE',
            'R_Wrist_JOINT_ANGLE']
        discard_index_list = []
        for i in marker_list:
            if i not in standard_marker_list:
                discard_index_list.extend(df_transposed[df_transposed['markers'] == i].index)
        df_transposed.drop(discard_index_list, axis = 0, inplace = True)
        discard_index_list = []
        # print(df_transposed)
        if stroke == True:
            pass
        else:
            toBeExcluded = []
            for subject in subject_list:
                for i in df_transposed["record"][df_transposed["subject"] == subject].unique():
                    if len(df_transposed["markers"][(df_transposed["subject"] == subject) & (df_transposed["record"] == i)].unique()) != len(standard_marker_list):
                        toBeExcluded.append(i)
            for records in toBeExcluded:
                discard_index_list.extend(df_transposed[df_transposed["record"] == records].index)
        if len(discard_index_list) != 0:
            df_transposed.drop(discard_index_list, axis = 0, inplace = True)
        DICT_repetition = collections.defaultdict()
        for subject in subject_list:
            repetition = int(sum((df_transposed["markers"] == 'L_Elbow_JOINT_ANGLE') & (df_transposed["subject"] == subject)) / 3)
            # repetition = sum((df_transposed["markers"] == marker) & (df_transposed["subject"] == subject)) / 3

            DICT_repetition[subject] = repetition

        subject_marker_dict = collections.defaultdict()
        repetition_list = []
        for subject in subject_list:
            subject_marker_list = df_transposed[df_transposed["subject"] == subject]["markers"].unique()
            subject_marker_dict[subject] = subject_marker_list
            # print(subject_marker_list)
            for i in range(len(subject_marker_list)):
                for j in range(DICT_repetition[subject]):
                    repetition_list.extend([j + 1, j + 1, j + 1])
        df_transposed.insert(3, 'repetition', repetition_list)
        subject_list = df_transposed["subject"].unique()




        return df_transposed, subject_list, subject_marker_dict, DICT_repetition

    def get_feature_dict(self, analyse_dict, axis_list):
        feature_dict = collections.defaultdict()
        for key in analyse_dict.keys():
            try:
                for i in analyse_dict[key]:
                    feature_dict[i] = axis_list[key]
            except:
                continue
        return feature_dict

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
                    JA = np.array(data[(data["subject"] == subject) & (data["markers"] == JA_to_analyse) & (data["axis"] == axis_to_analyse) & (data["repetition"] == i + 1)].drop(columns = ["subject", "markers", "repetition", "axis", "record"]).T).astype(float)
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
        try:
            JA_mean = np.mean(np.concatenate(list(subject_JA_repetition_dict.values())), axis = 0)
        except:
            print(subject_JA_repetition_dict.values())
            print(len(subject_JA_repetition_dict.values()))
        #     print(subject_list)
        #     print(subject_JA_repetition_dict)

        # print(JA_mean)

        return subject_JA_repetition_dict, JA_mean, n_sample

    def generate_norm_dist(self, normal_dist_data, show_graph = False):
        ''' to generate normal distribution for each point to analyse and return the confidence interval of 95%, show graph if needed'''
        confidence = st.t.interval(alpha=0.95, df=len(normal_dist_data)-1, loc=np.mean(normal_dist_data), scale=st.sem(normal_dist_data))
        if show_graph:
            count, bins, ignored = plt.hist(list(normal_dist_data), len(normal_dist_data))
            plt.show()
        return confidence

    def generate_score(self, normative_mean, stroke_mean, plot_graph = False):
        '''calculate dtw score between two mean graph, plot graph if needed to see the time warping relation'''
        score = dtw.distance(normative_mean[:,1], stroke_mean[:,1])
        if plot_graph:
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
            path = dtw.warping_path(normative_mean[:,1], stroke_mean[:,1])
            dtwvis.plot_warping(normative_mean[:,1], stroke_mean[:,1], path, fig = fig, axs = axs)
            plt.show()
        return score

    def main(self, normative_df, stroke_df, task, analyse_dict, data_point):
        print("Manipulating data...")
        data, subject_list, subject_marker_dict, repetition_dict = self.manipulate_data(normative_df, stroke = False)
        stroke_data, stroke_list, stroke_marker_dict, stroke_repetition_dict = self.manipulate_data(stroke_df, stroke = True)

        total_index = 0
        for key in task.keys():
            total_index = total_index + len(task[key])*len(analyse_dict[key])


        # consolidate end results
        feature_dict = self.get_feature_dict(analyse_dict, task)
        result = pd.DataFrame(columns = ['Joint Angle', 'Feature', 'Stroke Patient Score', 'Normative Confidence Interval'], index = [i for i in range(total_index)])
        rotation = {"X":"Flexion/Extension", "Y":"Abduction/Adduction", "Z":"External/Internal Rotation"}
        current_index = -1


        normal_dist_data = collections.defaultdict()

        feature_dict = self.get_feature_dict(analyse_dict, task)

        for each_feature in feature_dict.keys():
            print("joint angle:", each_feature)
            for each_axis in feature_dict[each_feature]:
                print("axis:", each_axis)
                current_index += 1
                analysing = rotation[each_axis]

                # do bootstraping to get normal distribution, confidence interval, dtw score for each point to analyse and each axis
                normative_JA, normative_JA_mean, normative_n_sample = self.get_joint_angle_dict(data, data_point, each_feature, each_axis, subject_list, repetition_dict, subject_marker_dict)
                stroke_JA, stroke_JA_mean, stroke_n_sample = self.get_joint_angle_dict(stroke_data, data_point, each_feature, each_axis, stroke_list, stroke_repetition_dict, stroke_marker_dict)
                normal_dist_list = []
                for subject in tqdm(subject_list):
                    individual_subject_list = [subject]
                    bootstrap_subject_list = [ x for x in subject_list if x != subject]
                    bootstrap_JA, bootstrap_JA_mean, bootstrap_n_sample = self.get_joint_angle_dict(data, data_point, each_feature, each_axis, bootstrap_subject_list, repetition_dict, subject_marker_dict)
                    individual_JA, individual_JA_mean, individual_n_sample = self.get_joint_angle_dict(data, data_point, each_feature, each_axis, individual_subject_list, repetition_dict, subject_marker_dict)
                    individual_score = self.generate_score(bootstrap_JA_mean, individual_JA_mean)
                    normal_dist_list.append(float(individual_score))
                confidence_interval = self.generate_norm_dist(normal_dist_list, show_graph=False)
                stroke_score = self.generate_score(normative_JA_mean, stroke_JA_mean, plot_graph = False)

                each_row = {}
                each_row["Joint Angle"] = each_feature
                each_row["Feature"] = analysing
                each_row["Stroke Patient Score"] = stroke_score
                each_row["Normative Confidence Interval"] = confidence_interval
                normal_dist_data['normative_'+str(each_feature) + '_' + str(each_axis)] = normal_dist_list
                normal_dist_data['stroke_'+str(each_feature)+'_'+str(each_axis)] = stroke_score

                sd = np.std(np.array(normal_dist_data['normative_'+str(each_feature) + '_' + str(each_axis)]))
                each_row["std of normative"] = sd

                mean = np.mean(np.array(normal_dist_data['normative_'+str(each_feature) + '_' + str(each_axis)]))
                each_row["mean of normative"] = mean

                temp_list = normal_dist_data['normative_'+str(each_feature) + '_' + str(each_axis)] + [normal_dist_data['stroke_'+str(each_feature)+'_'+str(each_axis)]]
                zscore_list = stats.zscore(temp_list)
                each_row["stroke z-score"] = zscore_list[-1]
                result.loc[current_index] = pd.Series(each_row)

        return data, result, normal_dist_data

    def upper_threshold(self, result_df):
        upper_threshold_list = []
        for index in range(len(result_df)):
            if (int(result_df["Stroke Patient Score"][index]) > result_df["Normative Confidence Interval"][index][0]) == True and (int(result_df["Stroke Patient Score"][index]) < result_df["Normative Confidence Interval"][index][1]) == True:
                upper_threshold_list.append(1)
                # result_df["Drop (Upper threshold)"][index] = 1
            else:
                upper_threshold_list.append(0)
                # result_df["Drop (Upper threshold)"][index] = 0
        result_df["Drop (Upper threshold)"] = upper_threshold_list
        return result_df

    def get_norm_dist(self, results, normal_dist_data):
        mean_list = []
        sd_list = []
        zscore_list = []
        for index in range(len(results)):
            if results["Feature"][index] == "Flexion/Extension":
                axis = "X"
            elif results["Feature"][index] == "Abduction/Adduction":
                axis = "Y"
            elif results["Feature"][index] == "External/Internal Rotation":
                axis = "Z"
                
            mean = np.mean(np.array(normal_dist_data['normative_'+str(results["Joint Angle"][index]) + '_' + str(axis)]))
            mean_list.append(mean)
            sd = np.std(np.array(normal_dist_data['normative_'+str(results["Joint Angle"][index]) + '_' + str(axis)]))
            sd_list.append(sd)
            zscores = (normal_dist_data['stroke_'+str(results["Joint Angle"][index]) + '_' + str(axis)] - mean) / sd
            zscore_list.append(zscores)

        results["Normative Mean"] = mean_list
        results["Normative SD"] = sd_list
        results["Stroke Z-Score"] = zscore_list

        return results



    def get_trial_ROM(self, df, feature_dict, first_result):
        df["feature"] = df["markers"] + '_'+ df["axis"]

        features_list = [i+'_'+j for i, k in feature_dict.items() for j in k]
        relevant_df = df[df["feature"].isin(features_list)]
        trial_rom_list = []
        # print(relevant_df)
        for i in range(len(relevant_df)):
            trial_rom_list.append(max([float(k) for k in relevant_df.iloc[i,5:-1]]) - min([float(k) for k in relevant_df.iloc[i,5:-1]]))
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
            # try:
            # print(average_list)
            resorted_average_list.append(average_list[first_result["Joint Angle"][i] + '_' + axis])
            # except:
        first_result["Trial ROM"] = resorted_average_list
        return relevant_df, features_list, first_result

    def get_significance_vector(self, result_table, feature_dict, analyse_dict):
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
            rom_list.append(self.ROM_dict[rangeOfMotion])
            significancy = (result_table["Trial ROM"][index]) / (int(self.ROM_dict[rangeOfMotion][1]) - int(self.ROM_dict[rangeOfMotion][0]))
            significancy_list.append(float(significancy))
        # print(significancy_list)
        result_table["ROM"] = rom_list
        result_table["weight"] = significancy_list

        
        return result_table




class FileDialogExample(QWidget):
    def __init__(self):
        super().__init__()

        self.button_selection_flag = False

        self.button_group = QButtonGroup()
        self.results_text = ""

        self.left_button = QPushButton("Left Side Affected")
        self.right_button = QPushButton("Right Side Affected")

        self.left_button.setCheckable(True)
        self.right_button.setCheckable(True)

        self.button_group.addButton(self.left_button, 1)
        self.button_group.addButton(self.right_button, 2)

        self.button_group.buttonClicked[int].connect(self.process_selected)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.left_button)
        button_layout.addWidget(self.right_button)
        self.layout = QVBoxLayout()
        self.layout.addLayout(button_layout)


        self.buttons = []
        self.labels = []

        self.button_name = [
            "towel_JA_2pick1_bwd_NORMATIVE",
            "towel_JA_4Pick2_cross_NORMATIVE",
            "grasp_JA_2block_shelf_NORMATIVE",
            "lateral_JA_1start_cross_NORMATIVE",
            "mouth_JA_1start_mouth_NORMATIVE",
            "key_JA_3turn_return_NORMATIVE",
            "key_JA_2touchkey_turn_NORMATIVE",

            "towel_JA_2pick1_bwd_STROKE",
            "towel_JA_4Pick2_cross_STROKE",
            "grasp_JA_2block_shelf_STROKE",
            "lateral_JA_1start_cross_STROKE",
            "mouth_JA_1start_mouth_STROKE",
            "key_JA_3turn_return_STROKE",
            "key_JA_2touchkey_turn_STROKE"
            ]

        self.num_buttons = len(self.button_name)  # Change this to the desired number of buttons

        for i in range(self.num_buttons):
            button = QPushButton(self.button_name[i])
            label = QLabel()
            self.layout.addWidget(button)
            self.layout.addWidget(label)

            self.buttons.append(button)
            self.labels.append(label)

            button.clicked.connect(lambda _, i=i: self.showDialog(i))  # Pass the index as a lambda parameter

        self.setLayout(self.layout)

        self.selected_file_paths = []

        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submitFiles)
        self.layout.addWidget(submit_button)

    def show_results(self, results):
        self.result_window = QMainWindow()
        result_widget = QTextEdit()
        result_widget.setPlainText(results)
        result_widget.setAlignment(Qt.AlignCenter)  # Center-align the text in the QTextEdit
        self.result_window.setCentralWidget(result_widget)
        self.result_window.setWindowTitle('Results')
        self.result_window.setGeometry(100, 100, 600, 400)  # Adjust the window size as needed
        self.result_window.show()

    def update_results_text(self, text):
        self.results_text += text

    def process_selected(self, button_id):
        if button_id == 1:
            self.affected_side = "Left"
            self.button_selection_flag = True
            self.right_button.setChecked(False)


        elif button_id == 2:
            self.affected_side = "Right"
            self.button_selection_flag = True
            self.left_button.setChecked(False)


    def showDialog(self, index):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        if ex.button_selection_flag == False:
            QMessageBox.warning(self, "Warning", "Affected side not chosen")
        else:
            file_dialog.setWindowTitle(f"Open {self.affected_side}_{self.button_name[index]}")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
        

            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()
                if selected_file:
                    self.selected_file_paths.append(selected_file[0])
                    self.labels[index].setText(selected_file[0])
                    print("checking file...")
                    with open(selected_file[0], 'r', newline='') as csvfile:
                        csvreader = csv.reader(csvfile, delimiter='\t')
                        # Read the first row (header) if it exists
                        header = next(csvreader, None)
                        # Check if the header is empty or contains only whitespace
                        if header and not any(cell.strip() for cell in header):
                            QMessageBox.warning(self, "Warning", "The selected file is empty!")
    

    
        
    def submitFiles(self):
        final_score_dict = collections.defaultdict()
        analysis_function = BCI_analysis(ex.affected_side)

        number_of_stroke_files = int(ex.num_buttons/2)
        for i in range(0, number_of_stroke_files):
            task = ex.button_name[i]
            last_underscore = task.rfind("_")
            task_name = task[:last_underscore]
            print("analysing", task_name)
            feature = analysis_function.get_feature_dict(analysis_function.analyse_dict, analysis_function.task_dict[task_name])
            # print(feature)
            normative = pd.read_csv(self.selected_file_paths[i], sep = '\t')
            stroke = pd.read_csv(self.selected_file_paths[i+number_of_stroke_files], sep = '\t')
            # stroke = pd.read_csv(self.selected_file_paths[i+7], sep = '\t')
            print(analysis_function.task_dict[task_name])
            df, dtw_result, norm_dist = analysis_function.main(normative, stroke, analysis_function.task_dict[task_name], analysis_function.analyse_dict, analysis_function.data_point)
            upper_threshold_result = analysis_function.upper_threshold(dtw_result)
            _, _, ROM_result = analysis_function.get_trial_ROM(df, feature, upper_threshold_result)
            significant_vector_result = analysis_function.get_significance_vector(ROM_result, feature, analysis_function.analyse_dict)
            norm_dist_result = analysis_function.get_norm_dist(significant_vector_result, norm_dist)

            if sum(norm_dist_result["Drop (Upper threshold)"]) != len(norm_dist_result["Drop (Upper threshold)"]):
                final_score = sum(norm_dist_result["weight"] * abs(norm_dist_result["Stroke Z-Score"]))
                final_score_dict[analysis_function.NHG_task_mapping[task_name]] = final_score
            else:
                final_score_dict[analysis_function.NHG_task_mapping[task_name]] = 0
    

        
            descending_final_score_dict = OrderedDict(sorted(final_score_dict.items(), key=lambda item: item[1], reverse=False))
            for task, result in descending_final_score_dict.items():
                if result == 0:
                    descending_final_score_dict[task] = "Pass"
            results_df = pd.DataFrame(list(descending_final_score_dict.items()), columns=["Task", "Result"])


            self.update_results_text("Processing Results:\n")
            self.update_results_text(norm_dist_result.to_string(index=False) + "\n\n")

        # Append the results to the results text
        self.update_results_text("Final Scores:\n")
        self.update_results_text(results_df.to_string(index=False) + "\n\n")

        # Display the results in a separate window
        self.show_results(self.results_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ex = FileDialogExample()
    window.setCentralWidget(ex)
    window.setWindowTitle('BCI Project')
    window.setGeometry(100, 100, 400, 300)  # Adjust the window size as needed
    window.show()

    sys.exit(app.exec_())
