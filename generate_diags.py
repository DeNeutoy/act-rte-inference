

import pickle
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import pandas as pd

def convert_to_dataframe(data):

    prem_max = np.expand_dims(np.max(data["premise_attention"],1),1)
    hyp_max = np.expand_dims(np.max(data["hypothesis_attention"],1),1)
    prem = pd.DataFrame(data["premise_attention"]/prem_max ,columns=data["premise"])
    hyp = pd.DataFrame(data["hypothesis_attention"]/hyp_max, columns=data["hypothesis"])

    return prem, hyp
with open("/Users/markneumann/Documents/Machine_Learning/act-rte-inference/weights/processed_data_test.pkl", "rb") as file:

    data = pickle.load(file)

for i in range(len(data)):

    # 9723, 156 is a good example
    # 334 for demonstrating weighting on incorrect examples
    # 4763 clear multi stage alignments
    # 9311 for multi-resolution
    # 4, good for concentration
    # 56 GREAT - co-reference resolution

    if i in [235, 56, 3454, 3565]:
        print(i,len(data[i]["premise_attention"]), len(data[i]["hypothesis_attention"]))
        print(data[i])

        fig= plt.figure()
        axs = gs.GridSpec(2,15)
        prem, hyp = convert_to_dataframe(data[i])
        max_val = max(prem.values.max(), hyp.values.max())
        max_dist = max(data[i]["act_probs"])
        ax = fig.add_subplot(axs[0,0:13])
        ax.set_title("Hypothesis")
        seaborn.heatmap(hyp, vmin=0.0, vmax=1.0)

        ax = fig.add_subplot(axs[1,0:13])
        ax.set_title("Premise")
        seaborn.heatmap(prem, vmin=0.0, vmax=1.0)
        ax = fig.add_subplot(axs[:,14])
        ax.set_title("ACT halting probs(weights)")
        seaborn.heatmap(np.expand_dims(data[i]["act_probs"],1),annot=True)
        plt.gca()
        plt.show()