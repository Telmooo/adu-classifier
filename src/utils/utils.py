from typing import (
    Callable,
    Optional,
    Union
)

import os
from pathlib import Path

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

"""
Pandas IO
"""
def read_csv(directory : str, filename : str, sep : str =',', **kwargs) -> pd.DataFrame:
    """
    Read CSV file into a Pandas dataframe
    """
    return pd.read_csv(Path(directory, filename), sep=sep, **kwargs)

def write_csv(df : pd.DataFrame, directory : str, filename : str, sep : str =',', write_index : bool =False) -> None:
    """
    Write Pandas dataframe into a CSV file
    Directory is created if doesn't exist
    """
    os.makedirs(directory, exist_ok=True)
    df.to_csv(Path(directory, filename), sep=sep, index=write_index)

def read_excel(directory : str, filename : str, **kwargs) -> pd.DataFrame:
    """
    Read Excel file into a Pandas dataframe
    """
    return pd.read_excel(Path(directory + '/' + filename), **kwargs)

"""
Utilities
"""
def convert_date(df : pd.DataFrame, date_column : str, format : str ="%Y-%m-%d %H:%M:%S", preprocess : Optional[Callable] = None) -> pd.DataFrame:
    """
    Convert column of dataframe given into a datetime object given the format and optionally pre-processing

    """
    out_df = df.copy()
    if preprocess:
        out_df[date_column] = out_df[date_column].apply(preprocess)
    out_df[date_column] = pd.to_datetime(out_df[date_column], format=format)
    return out_df

"""
Plotting & Visualisation
"""

def get_confusion_matrix(Y_true : np.array, Y_pred : np.array, directory : str, filename : str, **kwargs) -> None:
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(Y_true, Y_pred, ax=ax, **kwargs)
    os.makedirs(directory, exist_ok=True)
    fig.savefig(Path(directory, filename + '_ROC_CURVE.png'), format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.clf()

"""
Data Analysis
"""

def get_null_description(df : pd.DataFrame) -> pd.DataFrame:
    return df.isnull().agg(['count', 'sum', 'mean'])\
                        .rename(index={
                            'count' : 'total_rows',
                            'sum' : 'null_rows',
                            'mean' : 'ratio'
                        })

