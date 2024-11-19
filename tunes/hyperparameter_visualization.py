import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates

def find_best_params(data, metric_column='MRR'):
    """
    Finds the row in the dataset with the best value for the specified metric.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        metric_column (str): The column name of the metric to maximize (default: 'MRR').

    Returns:
        pd.Series: The row with the best metric value.
    """
    best_row = data.loc[data[metric_column].idxmax()]
    return best_row

def calculate_correlation_with_metric(data, metric_column='MRR'):
    """
    Calculates the correlation of all numeric columns with the specified metric column.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        metric_column (str): The column name of the metric to calculate correlations against (default: 'MRR').

    Returns:
        pd.Series: A sorted series of correlation values with the specified metric column.
    """
    correlations = data.corr(numeric_only=True)[metric_column].sort_values(ascending=False)
    return correlations

# Load the dataset
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_test/TAPE/tunes/'
group1 = ['groups_cora_NCN_mpnet.csv', 'groups_cora_NCN_MiniLM.csv', 'groups_cora_NCN_e5-large.csv']
group2 = ['group2_cora_NCN_mpnet.csv', 'group2_cora_NCN_MiniLM.csv']

for f in group2:
    data = pd.read_csv(f'{root}{f}')


    correlation_with_mrr = calculate_correlation_with_metric(data, metric_column='MRR')
    print(correlation_with_mrr)


    best_params = find_best_params(data, metric_column='MRR')
    print(best_params[:5], best_params['MRR'])
