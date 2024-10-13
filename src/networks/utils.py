import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, f1_score
import torch

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config():
    with open(os.path.join(os.getcwd(), "config.yaml"), "r") as config:
        args = AttributeDict(yaml.safe_load(config))
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    return args

def accuracy(pred_y, y):
    """Calculate accuracy"""
    return ((pred_y == y).sum() / len(y)).item()

def compute_metrics(model, name, data, df):

  _, y_predicted = model((data.x, data.edge_index))[0].to("cpu").max(dim=1)
  data = data.to("cpu")

  prec_ill,rec_ill,f1_ill,_ = precision_recall_fscore_support(data.y[data.test_mask], y_predicted[data.test_mask], average='binary', pos_label=0)
  f1_micro = f1_score(data.y[data.test_mask], y_predicted[data.test_mask], average='micro')

  m = {'model': name, 'Precision': np.round(prec_ill,3), 'Recall': np.round(rec_ill,3), 'F1': np.round(f1_ill,3),
   'F1 Micro AVG':np.round(f1_micro,3)}

  return m

def plot_results(df):
    labels = df['model'].to_numpy()
    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    f1_micro = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))
    width = 0.15
    _, ax = plt.subplots(figsize=(20, 7))

    ax.bar(x - width/2, precision, width, label='Precision', color='#83f27b')
    ax.bar(x + width/2, recall, width, label='Recall', color='#f27b83')
    ax.bar(x - (3/2) * width, f1, width, label='F1', color='#f2b37b')
    ax.bar(x + (3/2) * width, f1_micro, width, label='Micro AVG F1', color='#7b8bf2')

    ax.set_ylabel('Value')
    ax.set_title('Metrics for illicit class')

    # Rotate the x-tick labels vertically (90 degrees)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)  # Rotate by 90 degrees

    ax.set_yticks(np.arange(0, 1, 0.05))
    ax.legend(loc="lower left")

    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent label cut-off
    plt.show()



def aggregate_plot(df):

    labels = df['model'].to_numpy()

    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    maf1 = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))  # the label locations
    width = 0.55  # the width of the bars
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.bar(x, f1, width, label='F1 Score',color='#f2b37b')
    ax.bar(x , maf1, width, label='M.A. F1 Score',color='#7b8bf2',bottom=f1)
    ax.bar(x, precision, width, label='Precision',color='#83f27b',bottom=maf1 + f1)
    ax.bar(x, recall, width, label='Recall',color='#f27b83',bottom=maf1 + f1 + precision)

    ax.set_ylabel('value 0-1')
    ax.set_title('Final metrics by classifier')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,4,0.1))
    ax.set_xticklabels(labels=labels)
    ax.legend()

    plt.xticks(rotation=90)
    plt.grid(True)
    fig.tight_layout()
    plt.show()

class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__