from ensurepip import bootstrap
import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as st
from tqdm import tqdm
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis



def manipulate_data(raw, marker):
    '''
    first output: get standardized data structure; 
    second output: list of subject included in the study;
    third output: unique list of kinematics/kinetics/markers etc for each subject;
    fourth output: number of repetition for each subject
    '''

    # standardized headers and index
    raw.drop([1,2], inplace = True)
    subject = [i[0:5] for i in raw.columns]
    subject[0] = "subject"
    raw.loc[-1] = subject
    raw = raw.sort_index()
    raw.reset_index(inplace=True)
    raw.drop(columns=["index"], inplace = True)
    raw.rename(columns = {raw.columns[0] : "video"}, inplace = True)
    raw["video"][1] = "markers"
    raw["video"][2] = "axis"

    # transpose into time series structure
    raw_transposed = raw.T
    raw_transposed.columns = raw_transposed.iloc[0]
    raw_transposed.drop('video', axis = 0, inplace = True)

    # get list of subjects involved
    subject_list = raw_transposed["subject"].unique()

    # get unique list of markers/derived data
    marker_list = raw_transposed["markers"].unique()

    # calculate repetition for each subject
    DICT_repetition = collections.defaultdict()
    for subject in subject_list:
        repetition = int(sum((raw_transposed["markers"] == marker) & (raw_transposed["subject"] == subject)) / 3)
        DICT_repetition[subject] = repetition
    
    # standard markers/derived data respective study needs
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

    # discard the rest if not needed
    discard_index_list = []
    for i in marker_list:
        if i not in standard_marker_list:
            discard_index_list.extend(raw_transposed[raw_transposed['markers'] == i].index)
    raw_transposed.drop(discard_index_list, axis = 0, inplace = True)
    
    # calculate repetition in each kinematics/kinetics/markers etc (different from repetition in terms of subject just in case not all subject has the features)
    subject_marker_dict = collections.defaultdict()
    repetition_list = []
    for subject in subject_list:
        subject_marker_list = raw_transposed[raw_transposed["subject"] == subject]["markers"].unique()
        subject_marker_dict[subject] = subject_marker_list
        for i in range(len(subject_marker_list)):
            for j in range(DICT_repetition[subject]):
                repetition_list.extend([j + 1, j + 1, j + 1])
    raw_transposed.insert(3, 'repetition', repetition_list)

    return raw_transposed, subject_list, subject_marker_dict, DICT_repetition



def get_coordinates_dict(data, data_point, point_to_analyse, subject_list, repetition_dict, subject_marker_dict):
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





def get_joint_angle_dict(data, data_point, JA_to_analyse, axis_to_analyse, subject_list, repetition_dict, subject_marker_dict):
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


def pca_bandwidth(coordinates):
    ''' calculate the max and min bandwidth for trajectory (not yet in use)'''
    coordinates = coordinates.reshape(-1,2)
    X_std = StandardScaler().fit_transform(coordinates)
    pca = PCA(n_components=2)
    pca.fit(X_std)
    X_pca = pca.fit_transform(X_std)
    min_bandwidth = min(X_pca[:, 1])
    max_bandwidth = max(X_pca[:, 1])
    return X_pca, min_bandwidth, max_bandwidth

def plot_graph(data, graph, specific_subject = False):
    '''to plot out graph if needed, able to specific subject to plot if needed'''

    # if is numpy, will plot out mean data
    if type(data).__module__ == np.__name__:
        graph.scatter(data[:, 0], data[:, 1], alpha=0.8, color = 'red', s = 1)

    # if is dict, will plot out individual trajectory
    elif type(data) is collections.defaultdict:
        if specific_subject:
            specific_subset = {key: data[key] for key in specific_subject}
            data = np.concatenate(list(specific_subset.values()))
        else:
            data = np.concatenate(list(data.values()))
        for time_series in data:
            graph.scatter(time_series[:, 0], time_series[:, 1], alpha=0.8, color = 'blue', s = 1)


def generate_norm_dist(normal_dist_data, show_graph = False):
    ''' to generate normal distribution for each point to analyse and return the confidence interval of 95%, show graph if needed'''
    confidence = st.t.interval(alpha=0.95, df=len(normal_dist_data)-1, loc=np.mean(list(normal_dist_data.values())), scale=st.sem(list(normal_dist_data.values())))
    if show_graph:
        count, bins, ignored = plt.hist(list(normal_dist_data.values()), len(normal_dist_data))
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


if __name__ == "__main__":

    # load data
    df = pd.read_csv(r"Z:\DataCollection\BCISoftRoboticsGloveIntervention\Stroke\Miscellaneous\v3d_workspaces\1. EXPORTS\EXPORT_mouth\LeftJA_1Start_Mouth.txt", sep = '\t')
    stroke = pd.read_csv(r"C:\Users\Administrator\Documents\gitHub\BCI\data\stroke_mouth_L_start_mouth", sep = '\t')

    # input which point to analyse and which axis to analyse
    analyse_list = ["R_Elbow_JOINT_ANGLE", "R_Shoulder_JOINT_ANGLE", "R_Wrist_JOINT_ANGLE" ]
    axis_list = ["X", "Y", "Z"]

    # if all data has already been set to a standardize normalize point, otherwise, remove this and change manually in each 'get_joint_angle_dict/get_coordinate_dict' functions
    data_point = 300

    # start standardizing all data
    data, subject_list, subject_marker_dict, repetition_dict = manipulate_data(df, "L_Elbow_JOINT_ANGLE")
    stroke_data, stroke_list, stroke_marker_dict, stroke_repetition_dict = manipulate_data(stroke, "L_Elbow_JOINT_ANGLE")

    # consolidate end results
    total_index = len(analyse_list) * len(axis_list)
    result = pd.DataFrame(columns = ['Joint Angle', 'Feature', 'Stroke Patient Score', 'Normative Confidence Interval'], index = [i for i in range(total_index)])
    rotation = {"X":"Flexion/Extension", "Y":"Abduction/Adduction", "Z":"External/Internal Rotation"}
    current_index = -1


    for each_feature in analyse_list:
        for each_axis in axis_list:
            current_index += 1
            analysing = rotation[each_axis]

            # do bootstraping to get normal distribution, confidence interval, dtw score for each point to analyse and each axis
            JA, JA_mean, n_sample = get_joint_angle_dict(data, data_point, each_feature, each_axis, subject_list, repetition_dict, subject_marker_dict)
            stroke_JA, stroke_JA_mean, stroke_n_sample = get_joint_angle_dict(stroke_data, data_point, each_feature, each_axis, stroke_list, stroke_repetition_dict, stroke_marker_dict)
            normal_dist_data = collections.defaultdict()
            for subject in subject_list:
                individual_subject_list = [subject]
                bootstrap_subject_list = [ x for x in subject_list if x != subject]
                bootstrap_JA, bootstrap_JA_mean, bootstrap_n_sample = get_joint_angle_dict(data, data_point, each_feature, each_axis, bootstrap_subject_list, repetition_dict, subject_marker_dict)
                individual_JA, individual_JA_mean, individual_n_sample = get_joint_angle_dict(data, data_point, each_feature, each_axis, individual_subject_list, repetition_dict, subject_marker_dict)
                individual_score = generate_score(bootstrap_JA_mean, individual_JA_mean)
                normal_dist_data[subject] = individual_score
                normal_dist_data
            confidence_interval = generate_norm_dist(normal_dist_data, show_graph=True)
            stroke_score = generate_score(JA_mean, stroke_JA_mean)
            

            # un-comment this is wants to plot out raw trajectories of all data
            # fig=plt.figure()
            # graph = fig.add_subplot(111)
            # plot_graph(stroke_JA_mean, graph)
            # # plot_graph(JA_mean, graph)
            # graph.set_xlabel('x')
            # graph.set_ylabel('y')
            # graph.set_title('test')
            # graph.axis('equal')
            # plt.autoscale()
            # plt.show()
            



            # compute results for each point to analyse and each axis
            each_row = {}
            each_row["Joint Angle"] = each_feature
            each_row["Feature"] = analysing
            each_row["Stroke Patient Score"] = stroke_score
            each_row["Normative Confidence Interval"] = confidence_interval
            result.loc[current_index] = pd.Series(each_row)


    # print("score:", stroke_score)
    # print("confidence interval:", confidence_interval)
    # print(normal_dist_data)
    # print(result)

    
    # coordinates, coordinates_mean, n_sample = get_coordinates_dict(data, "LACR", subject_list, repetition_dict, subject_marker_dict)
    # coordinates_pca, min_pca, max_pca = pca_bandwidth(coordinates)
    
    # to_plot = []
    # for i in distribution.keys():
    #     if distribution[i] <10:
    #         to_plot.append(i)
    # print(to_plot)



