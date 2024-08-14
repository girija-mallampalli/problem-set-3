'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv("model_predictions.csv")
    genres_df = pd.read_csv("genres.csv")
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = genres_df.columns.tolist()
    
    genre_true_counts = genres_df.sum().to_dict()
    genre_tp_counts = {}
    genre_fp_counts = {}

    for genre in genre_list:
        genre_tp_counts[genre] = ((model_pred_df[genre] == 1) & (genres_df[genre] == 1)).sum()
        genre_fp_counts[genre] = ((model_pred_df[genre] == 1) & (genres_df[genre] == 0)).sum()
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
