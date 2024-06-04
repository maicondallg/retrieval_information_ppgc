import numpy as np
from sklearn.metrics import confusion_matrix

def calcula_precision(y_pred, y_true, precisions):
    return {f'P@{prec}': ((y_pred[:prec]==y_true).sum())/prec for prec in precisions}

def calcula_map(y_pred, y_true):
    recall = np.where(y_pred == y_true)[0].shape[0]
    return np.sum(list(calcula_precision(y_pred, y_true, precisions=[int(idx*recall/10) for idx in range(1, 11)]).values()))
        
def calcula_matriz_confusao(y_pred, y_true, precisions, labels):
    return {prec: confusion_matrix(y_true=[y_true]*prec, y_pred=y_pred[:prec], labels=labels) for prec in precisions}

def calcula_measures(y_pred, y_true, precisions):
    map_m = calcula_map(y_pred, y_true)
    precisions.remove('map')
    measures = calcula_precision(y_pred, y_true, precisions=precisions)
    measures['MAP'] = map_m/10
    return measures

def fill_diagonal(source_array, diagonal):
    copy = source_array.copy()
    np.fill_diagonal(copy, diagonal)
    return copy