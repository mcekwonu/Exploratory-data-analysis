import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, entropy
from collections import Counter


class DataExploration:
    """Exploratory Data Analysis class for preprocessing continuous and categorical
    data.
    The categorical data columns are set to category type and encoded as int64.And
    the processed dataframe is saved with continuous data and categorical distinctively
    separated for further use as input feed for deep learning networks.

    Parameters:
        data_path (str): Input data directory as either `.csv` or `.xlsx`.
        save_dir (str): Output directory to save processed data.
        print_summary (bool): Display summary of data type, number of unique values.
    """

    def __init__(self,
                 data_path,
                 save_dir,
                 print_summary=False,
                 ):

        self.data_path = data_path
        self.save_dir = save_dir
        self.print_summary = print_summary
        self.filename = os.path.basename(data_path).split('.')[0]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        if self.data_path.endswith('csv'):
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_excel(data_path)

        # run data preprocessing, encode categorical data and save output
        self.preprocess

        if self.print_summary:
            self.print_data_summary

    def __str__(self):
        return f'{self.df.head()}'

    @property
    def preprocess(self):
        self.set_categorical_data
        self.get_categorical_classes
        self.plot_heatmap

    @property
    def set_categorical_data(self):
        df = self.df.copy()
        for column in df.columns:
            if df[column].dtypes == 'object':
                df[column] = df[column].astype('category')
                df[column] = df[column].cat.codes
        self._save_data(df)

    @property
    def get_categorical_classes(self):
        df = self.df.copy()
        categorical_classes = {}
        numerical_columns = []
        for column in df.columns:
            if df[column].dtypes == 'object':
                cat_col = df[column].astype('category')
                categorical_classes[column.lower()] = {idx: cat_feature
                                                       for cat_feature, idx in zip(cat_col, cat_col.cat.codes)
                                                       }
            else:
                numerical_columns.append(column)
        np.save(f'{self.save_dir}/{self.filename}_numerical_col', numerical_columns)
        np.savez(f'{self.save_dir}/{self.filename}_categorical_col', **categorical_classes)

    def _save_data(self, dataframe):
        dataframe.to_csv(f'{self.save_dir}/{self.filename}.csv', index=False)

    @property
    def print_data_summary(self):
        for column in self.df.columns:
            unique_values = self.df[column].unique()

            print(f'Statistics for column: {column}')
            print(f'Unique values:\n {unique_values}')
            print(f'Number of unique values: {len(unique_values)}')
            print(f'Number of NAN values: {self.df[column].isna().sum()}')
            print(f'dtype: {self.df[column].dtypes}')
            print('_' * 70)

    @property
    def plot_heatmap(self):
        plt.subplots(figsize=(18, 18))

        corr = self.df.corr(method='spearman')
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.savefig(f'{self.save_dir}/data_heatmap.png', dpi=600)
        plt.tight_layout()

    def merge_data(self, *dataframes):
        return pd.concat(*dataframes, axis=0, ignore_index=True)

    def cramers_v(self, x, y):
        """Computes Cramer's V statistics for categorical-categorical association
        """
        confusion_matrix = pd.crosstab(x, y)
        chi_2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        phi_2 = chi_2 / n
        phi_corr = max(0, phi_2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi_corr / min(k_corr - 1, r_corr - 1))
        return result

    def conditional_entropy(self, x, y):
        """Computes the conditional entropy of x given y: S(x|y)
        """
        y_counter = Counter(y)
        xy_counter = Counter((list(zip(x, y))))
        total_occurences = sum(y_counter.values())
        entropy = 0.0

        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurences
            p_y = y_counter[xy[1]] / total_occurences
            entropy += p_xy * np.log(p_y / p_xy)
        return entropy

    def theils_u(self, x, y):
        """Computes Theil's U statistics (Uncertainty coefficient) for
        categorical-categorical association.
        """
        s_xy = self.conditional_entropy(x, y)
        x_counter = Counter(x)
        total_occurences = sum(x_counter.values())
        p_x = list(map(lambda n: n / total_occurences, x_counter.values()))
        s_x = entropy(p_x)

        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_xy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Data analysis of numerical and categorical data'
    )
    parser.add_argument('-i', '--data_path', type=str, required=True,
                        help='Input file data path (files with `.csv` or `.xlsx`)')
    parser.add_argument('-s', '--summary', default=False, type=bool,
                        help='print data summary.')
    parser.add_argument('-o', '--save_dir', type=str, required=True,
                        help='Output save directory')
    opt = parser.parse_args()

    eda = DataExploration(data_path=opt.data_path,
                          print_summary=opt.summary,
                          save_dir=opt.save_dir)
