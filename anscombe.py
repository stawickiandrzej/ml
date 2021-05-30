import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# defining lists of arguments for Anscombe's quartet
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]


def get_output_dir() -> str:
    """get_output_dir [creates and generates local path for output folder]

    Returns:
        [directory]: [directory for figure location]
    """
    cwd = os.getcwd()
    output = os.path.join(cwd, "output")
    os.makedirs(output, exist_ok=True)
    return output


def dict_anscombe(x: list, y1: list, y2: list, y3: list, x4: list, y4: list, output: str) -> pd.DataFrame:
    """dict_anscombe [generates and saves DataFrame of Anscombe quartet out of the lists of variables]

    Args:
        x (list): [variable x]
        y1 (list): [variable y1]
        y2 (list): [variable y2]
        y3 (list): [variable y3]
        x4 (list): [variable x4]
        y4 (list): [variable y4]
        output (str): [folder path]

    Returns:
        pd.DataFrame: [DataFrame of Anscombe quartet]
    """
    df_dict = {
        'I': x,
        'II': y1,
        'III': y2,
        'IV': y3,
        'V': x4,
        'VI': y4
    }
    anscombe_df = pd.DataFrame(df_dict)
    anscombe_df.to_csv(os.path.join(
        output, 'anscombe_q_data.csv'), index=False)
    return anscombe_df


def anscombe_plot(x: list, y1: list, y2: list, y3: list, x4: list, y4: list, output: str) -> plt:
    """anscombe_plot [generates and saves plot for Anscombe quartet]

    Args:
        x (list): [variable x]
        y1 (list): [variable y1]
        y2 (list): [variable y2]
        y3 (list): [variable y3]
        x4 (list): [variable x4]
        y4 (list): [variable y4]
        output (str): [folder path]
    """
    scatter_plot_df = anscombe_df.reset_index()
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6), )
    axs[0, 0].set(xlim=(0, 20), ylim=(2, 14))
    axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12))
    axs[0, 0].scatter(x, y1)
    axs[0, 0].title.set_text("Set I")
    axs[0, 1].scatter(x, y2)
    axs[0, 1].title.set_text("Set II")
    axs[1, 0].scatter(x, y3)
    axs[1, 0].title.set_text("Set III")
    axs[1, 1].scatter(x4, y4)
    axs[1, 1].title.set_text("Set IV")
    fig.tight_layout()
    plt.savefig(os.path.join(output, 'anscombe_q_plot.jpg'))


def statistics_frame(df: pd.DataFrame, output: str) -> pd.DataFrame:
    """statistics_frame [returns statistical data about the Anscomble quartet]

    Args:
        df (pd.DataFrame): [input Anscomble DataFrame]
        output (str): [calculated statistical values for the DataFrame]

    Returns:
        pd.DataFrame: [calculated statistical values for the DataFrame]
    """
    def statistics_math(x: list, y: list) -> list:
        """statistics_math [calculates statistical data for given params]

        Args:
            x (list): [x from DataFrame]
            y (list): [y from DataFrame]

        Returns:
            list: [list of calculated statistical data]
        """
        variance = round(np.var(x), 3)
        mean = round(np.mean(x), 3)
        std_dev = round(np.std(x), 3)
        corr = round(pearsonr(x, y)[0], 3)

        return [variance, mean, std_dev, corr]

    set_I = statistics_math(df["I"], df["II"])
    set_II = statistics_math(df["I"], df["III"])
    set_III = statistics_math(df["I"], df["IV"])
    set_IV = statistics_math(df["V"], df["VI"])

    columns = ["var", "mean", "std", "corr"]
    index = ["set_I", "set_II", "set_III", "set_IV"]
    data = [set_I, set_II, set_III, set_IV]
    calculated_stat_frame = pd.DataFrame(data, index=index, columns=columns)

    calculated_stat_frame.to_csv(os.path.join(
        output, "calculated_stat_frame.csv"), index_label="index")

    return calculated_stat_frame


if __name__ == '__main__':
    output = get_output_dir()
    anscombe_df = dict_anscombe(x, y1, y2, y3, x4, y4, output)
    anscombe_plot(x, y1, y2, y3, x4, y4, output)
    statistics_frame(anscombe_df, output)
