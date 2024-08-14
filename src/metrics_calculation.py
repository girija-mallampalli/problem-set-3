'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

   # Micro metrics
    tp_sum = sum(genre_tp_counts.values())
    fp_sum = sum(genre_fp_counts.values())
    fn_sum = sum([genre_true_counts[genre] - genre_tp_counts[genre] for genre in genre_list])

    micro_precision = tp_sum / (tp_sum + fp_sum)
    micro_recall = tp_sum / (tp_sum + fn_sum)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    
    # Macro metrics
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []
    
    for genre in genre_list:
        precision = genre_tp_counts[genre] / (genre_tp_counts[genre] + genre_fp_counts[genre]) if genre_tp_counts[genre] + genre_fp_counts[genre] > 0 else 0.0
        recall = genre_tp_counts[genre] / genre_true_counts[genre] if genre_true_counts[genre] > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        
        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)
    
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list


    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    y_true = model_pred_df[genre_list].values
    y_pred = model_pred_df[genre_list].values
    
    # Calculate macro and micro metrics using precision_recall_fscore_support
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
