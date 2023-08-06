import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Get relevant metrics from a history
def get_metrics(history):
    history = history.history
    acc = history['sparse_categorical_accuracy']
    val_acc = history['val_sparse_categorical_accuracy']
    return acc, val_acc


# Plot training and evaluation metrics given a dict of histories
def plot_train_eval(history_dict):
    metric_dict = {}

    for k, v in history_dict.items():
        acc, val_acc = get_metrics(v)
        metric_dict[f'{k} training acc'] = acc
        metric_dict[f'{k} eval acc'] = val_acc

    acc_plot = pd.DataFrame(metric_dict)

    acc_plot = sns.lineplot(data=acc_plot, markers=True)
    acc_plot.set_title('training vs evaluation accuracy')
    acc_plot.set_xlabel('epoch')
    acc_plot.set_ylabel('sparse_categorical_accuracy')
    plt.show()


# Plot for comparing the two student models