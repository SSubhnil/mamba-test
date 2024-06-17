import pandas as pd
import os
import matplotlib.pyplot as plt

# Set the working directory
os.chdir('/home/subhnils/Downloads')

# Read CSV files into pandas dataframes
mamba_best_medium = pd.read_csv('./MAMBA_CSV/Walker_Best_return_Medium.csv')
mamba_best_xsmall = pd.read_csv('./MAMBA_CSV/Walker_Best_return_xSmall.csv')
mamba_eval_medium = pd.read_csv('./MAMBA_CSV/Walker_Eval_return_Medium.csv')
mamba_eval_xsmall = pd.read_csv('./MAMBA_CSV/Walker_Eval_return_xSmall.csv')
mamba_train_medium = pd.read_csv('./MAMBA_CSV/Walker_Train_return_Medium.csv')
mamba_train_xsmall = pd.read_csv('./MAMBA_CSV/Walker_Train_return_xSmall.csv')

dreamer_RUN_small = pd.read_csv('./DREAMER_CSV/RUN_return_small.csv')
dreamer_WALK_small = pd.read_csv('./DREAMER_CSV/WALK_return_small.csv')
dreamer_RUN_step_small = pd.read_csv('./DREAMER_CSV/RUN_step_small.csv')

# Print column names to understand the structure of the data
print(mamba_train_medium.columns.values)
print(dreamer_WALK_small.columns.values)

# Plotting function for subplots
def plot_comparison_subplots(dataframes, y_columns, labels, titles, nrows, ncols, figsize=(15, 15)):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten(order='F')  # Flatten the 2D array of axes for easy iteration

    for ax, df, y_col, label, title in zip(axs, dataframes, y_columns, labels, titles):
        for d, y, l in zip(df, y_col, label):
            ax.plot(d[y], label=l)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust the height spacing between subplots
    plt.show()

# Example subplots with x-axis as 'Episode' or another column
# Adjust the x_column and y_columns as needed based on your data
dataframes = [
    [mamba_train_medium, dreamer_RUN_small],
    [mamba_train_medium, dreamer_RUN_step_small],
    [mamba_train_medium, dreamer_RUN_small],
    [mamba_train_medium, dreamer_WALK_small],
    [mamba_train_medium, dreamer_WALK_small],
    [mamba_train_medium, dreamer_WALK_small]
]

# x_column = 'Episode'  # Example x-axis column
y_columns = [
    ['RUN-vanilla-medium - train_return', 'RUN_vanilla_small'],
    ['RUN-step-medium - train_return__MAX', 'Harris_RUN_step_random_small - RUN_step_random_small/episode/sum_abs_reward'],
    ['RUN-swelling-medium - train_return', 'RUN_swelling_small'],
    ['WALK-vanilla-medium - train_return', 'Vanilla_small__MAX'],
    ['WALK-step-medium - train_return', 'Harris_WALK_step_random_small'],
    ['WALK-swelling-medium - train_return', 'Swelling_small_random_everything__MAX']
]

labels = [
    ['MAMBA', 'DreamerV3'],
    ['MAMBA', 'DreamerV3'],
    ['MAMBA', 'DreamerV3'],
    ['MAMBA', 'DreamerV3'],
    ['MAMBA', 'DreamerV3'],
    ['MAMBA', 'DreamerV3']
]

titles = [
    'RUN vanilla',
    'RUN step force',
    'RUN swelling force',
    'WALK vanilla',
    'WALK step force',
    'WALK swelling force'
]

plot_comparison_subplots(
    dataframes=dataframes,
    #x_column=x_column,
    y_columns=y_columns,
    labels=labels,
    titles=titles,
    nrows=3,  # Number of rows
    ncols=2,  # Number of columns
    figsize=(15, 15)
)
