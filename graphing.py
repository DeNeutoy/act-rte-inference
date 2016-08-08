

import seaborn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
"""ref:
    config, epoch,epoch_files,eval_config,
    test_acc,test_file,test_loss,test_step_mean,
    test_step_var,train_acc,train_loss,
    train_step_mean,train_step_var,val_acc,val_loss
    val_step_mean, val_step_var"""

def load_stats(directory):


    all_statistics = []
    best_test_acc = ("", 0.0, 0.0)
    for file in os.listdir(directory):
        stats = pickle.load(open(os.path.join(directory,file), "rb"))
        all_statistics.append(stats)
        if np.max(stats["val_acc"]) > best_test_acc[1]:
            best_test_acc = (file, np.max(stats["val_acc"]), stats["test_acc"])

    print("Best Val Acc: ", best_test_acc[1], "Test Acc: ", best_test_acc[2])
    print("Filename: ", best_test_acc[0])

    return all_statistics

def load_proccesed_data(directory):

    config_data = []
    for file in os.listdir(directory):

        conf, data = pickle.load(open(os.path.join(directory,file), "rb"))
        config_data.append((conf,data))

    return config_data



def single_mean_with_variance(stats, title, save=False):

    """ Plot the mean number of ACT steps for a single run
    with variance bounds above and below   """
    labels = ["Training Set", "Validation Set"]
    datasets = [("train_step_mean", "train_step_var"),("val_step_mean", "val_step_var")]
    leg = []
    fig = plt.figure()
    colours = cm.rainbow(np.linspace(0,1,2))

    for key, label, color in zip(datasets, labels, colours):
        means = np.array(stats[key[0]]) + 1.0
        vars = np.array(stats[key[1]])

        plotted, =plt.plot(stats["epoch"],means, label=label, color=color)
        leg.append(plotted)
        upper = means + np.sqrt(vars)
        lower = means - np.sqrt(vars)
        plt.fill_between(stats["epoch"],lower, upper, alpha=0.3, color=color)

    plt.legend(handles=leg)

    ax = plt.gca()
    ax.set_title(title + ": Step Penalty = " + str(stats["config"][0].step_penalty))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean ACT Steps")
    ax.set_ylim([1,6])
    if save:
        plt.savefig("mean_and_variance" + str(stats["config"][0].step_penalty) +".png" )
    else:
        plt.show()

def mean_average_steps(all_loaded_stats, title):

    """ Plot all runs lightly, with colours corresponding to different step penalties.
        Plot the mean per step_penalty in bold. """

    step_params = list(set([run["config"][0].step_penalty for run in all_loaded_stats]))
    colours = cm.rainbow(np.linspace(0,1,len(step_params)))
    fig = plt.figure()
    mean_dict = defaultdict(list)

    # plot all runs lightly and accumulate runs per step_penalty parameter
    for run in all_loaded_stats:

        c = colours[step_params.index(run["config"][0].step_penalty)]
        data = np.array(run["val_step_mean"]) + 1.0                 #TODO: fix this
        plt.plot(run["epoch"], data ,color=c ,alpha=0.2)
        mean_dict[run["config"][0].step_penalty].append(data)

    # now plot the mean values in bold
    for key, value in mean_dict.items():

        c = colours[step_params.index(key)]
        data = np.vstack(value).mean(0)
        plt.plot(loaded_stats[0]["epoch"],data, color=c, alpha=1.0, linewidth=2.0)

    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean ACT steps")
    plt.show()


def avg_acc_per_class_wrt_ponder_cost(all_processed_data):


    step_params = list(set([run["config"][0].step_penalty for run in loaded_stats]))
    step_penalty_dict = {x:defaultdict(list) for x in step_params}

    plotting_data = []

    for config, data in all_processed_data:

        for class_type in [0.0,1.0,2.0]:
            step_penalty_dict[config.step_penalty][class_type]\
                .append(sum([x["correct"] for x in data if x["class"]==class_type])
                        /len([x["correct"] for x in data if x["class"]==class_type]))

    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim(0.5,1)
    ind = np.arange(len(step_params))
    bar_width = 0.20
    width = 0.35

    plt.bar(ind, [np.mean(x[0.0]) for x in step_penalty_dict.values()],bar_width, color="r")
    plt.bar(ind+width, [np.mean(x[1.0]) for x in step_penalty_dict.values()],bar_width, color="b")
    plt.bar(ind+width+width, [np.mean(x[2.0]) for x in step_penalty_dict.values()], bar_width, color="g")

    ax.set_title("accuracy per class/step penalty")
    plt.show()

def sentence_length_vs_ponder_time(config,processed_data):

    plotting_data = []

    for  data in processed_data:

        steps = len(data["act_probs"])
        hyp_length = sum([1 for x in data["hypothesis"] if x!="PAD"])
        prem_length = sum([1 for x in data["premise"] if x!="PAD"])
        avg_length = (hyp_length + prem_length)/2
        correct = data["correct"]
        type_class = data["class"]
        plotting_data.append([steps, avg_length, correct, type_class])


    plotting_data = pd.DataFrame(np.vstack(plotting_data), columns=["steps", "avg_length", "correct", "class"])

    seaborn.violinplot(x="steps", y="avg_length",
                       hue="correct",split=True,
                       data=plotting_data, inner="quartile", scale="count")
    plt.show()
    # for class_type in [0.0,1.0,2.0]:
    #     fig = plt.figure()
    #     x_vals = [x[0] for x in plotting_data if (x[3]==class_type and x[2]==1.0)]
    #     y_vals = [x[1] for x in plotting_data if (x[3]==class_type and x[2]==1.0)]
    #     print("Class: ",class_type, "No. Correct: ", len(x_vals))
    #     plt.scatter(x_vals, y_vals,color="g")
    #
    #     x_vals = [x[0] for x in plotting_data if (x[3]==class_type and x[2]==0.0)]
    #     y_vals = [x[1] for x in plotting_data if (x[3]==class_type and x[2]==0.0)]
    #     print("Class: ",class_type, "No. Incorrect: ", len(x_vals))
    #
    #     plt.scatter(x_vals, y_vals,color="r")
    #
    #     ax = plt.gca()
    #     ax.set_xlabel("ACT Steps")
    #     ax.set_ylabel("avg hyp/premise length")
    #     ax.set_title("test_title")
    #     plt.show()

if __name__=="__main__":

    stats_path = "/Users/markneumann/Documents/Machine_Learning/" \
                "act-rte-inference/weights/all_stats/full_run_stats"
    processed_data_path = "/Users/markneumann/Documents/Machine_Learning/" \
                "act-rte-inference/weights/all_stats/full_run_data"

    best_stats_path = "/Users/markneumann/Documents/Machine_Learning/act-rte-inference/weights/" \
                      "all_stats/AIAA_lr_0.0001hid_256drop_0.8eps_0.01step_pen_0.01_stats.pkl"

    best_stats = pickle.load(open(best_stats_path, "rb"))
    loaded_stats = load_stats(stats_path)
    #loaded_processed_data = load_proccesed_data(processed_data_path)
    #mean_average_steps(loaded_stats, "test_title")
    #avg_acc_per_class_wrt_ponder_cost(loaded_processed_data)
    #sentence_length_vs_ponder_time(*loaded_processed_data[1])

    filtered_stats = [stat for stat in loaded_stats if
                      (stat["config"][0].hidden_size==256 and
                       stat["config"][0].learning_rate==0.0001 and
                       stat["config"][0].keep_prob==0.6)]

    for stat in filtered_stats:
        single_mean_with_variance(stat, "Mean(+/- SD) Number of ACT Steps per Epoch", save=True)