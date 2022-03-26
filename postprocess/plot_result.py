import matplotlib.pyplot as plt
import pandas as pd


def loss_metric_plot(history, epochs, img_path, plot_title, y_label):
    train_loss = history['train_loss']
    eval_precision = history['eval_precision']
    eval_recall = history['eval_recall']
    eval_f1 = history['eval_f1']
    steps = history['step']

    show_every = len(steps) // epochs
    sparse_epochs = [None] * len(steps)
    print(show_every)
    print(len(sparse_epochs))
    sparse_epochs[show_every-1::show_every] = list(range(1, epochs+1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(steps, train_loss, label='Training')
    axes[0].set_xticks(steps)
    axes[0].set_xticklabels(sparse_epochs)

    axes[0].legend(loc='upper right')
    # axes[0].set_xlabel('Epochs', fontsize='large')
    axes[0].set_ylabel(y_label, fontsize='large')
    # axes[0].grid()
    axes[1].plot(steps, eval_precision, label='Precision')
    axes[1].plot(steps, eval_recall, label='Recall')
    axes[1].plot(steps, eval_f1, label='F1-score')
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(sparse_epochs)
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Num of Epochs', fontsize='large')
    axes[1].set_ylabel('Eval Precision, Recall, F1-score', fontsize='large')
    # axes[1].grid()
    axes[0].set_title(plot_title, fontsize='x-large')
    #plt.show()
    plt.savefig(img_path)


# 绘图时请保证 checkpoints/eval_results.csv 存在，并且训练是完全完成的，没有中断
eval_results_df = pd.read_csv("../checkpoints/eval_results.csv")
loss_metric_plot(eval_results_df, 30, "./result.png", "Bert for DoctorKG - Performance", "loss")
