import numpy as np
import pandas as pd

def bootstrap_2_samples_median(data, n_times, n_samples_x, n_samples_y):
    means_x = np.zeros(n_times)
    means_y = np.zeros(n_times)
    for k in range(0,n_times):
        means_x[k] = np.median(np.random.choice(data, n_samples_x, replace=True))
        means_y[k] = np.median(np.random.choice(data, n_samples_y, replace=True))
    return means_x, means_y

def bootstrap_test_2_samples_median(data, X, Y, N=50000):
    means_x, means_y = bootstrap_2_samples_median(data, N, X.shape[0], Y.shape[0])
    return (means_y - means_x)-(np.median(Y)-np.median(X))

def bootstrap_test_2_samples_median_2_tailed(data, X, Y, N=50000):
    means_x, means_y = bootstrap_2_samples_median(data, N, X.shape[0], Y.shape[0])
    return np.abs(means_y - means_x) - np.abs(np.median(Y)-np.median(X))

def pvalue_ceiling(vector):
    p = (vector > 0).mean()
    return p if p else np.int64(1)/len(vector)

def matched_pairs_multiple_factors(match_data_original, query_set, query_columns, col_weights=None, distance_cutoff = 0.2, studentize_cols=False):
    match_data = match_data_original.copy()
    if studentize_cols:
        for column in query_columns:
            match_data[column] = studentize(match_data[column])
    if col_weights:
        assert len(col_weights) == len(query_columns)
        for i, column in enumerate(query_columns):
            match_data[column] *= col_weights[i]
    x = closest_L2_pairs(match_data.ix[match_data.gene_name.isin(query_set), query_columns].as_matrix(),
                 match_data.ix[~match_data.gene_name.isin(query_set), query_columns].as_matrix(),
                )
    
    mins = x.argmin(axis=1)
    used_elements = set()
    pairs = []
    
    for i in range(x.shape[0]):
        if mins[i] not in used_elements:
            pairs.append((i, mins[i], x[i, mins[i]]))
            used_elements.add(mins[i])
        else:
            j = 2
            next_best = np.argsort(x[i , :])
            while x[i, next_best[j]] in used_elements:
                j += 1
            pairs.append((i, next_best[j], x[i, next_best[j]]))
    
    ordered_pairs = pd.DataFrame(pairs)
    ordered_pairs.columns = ('query_index', 'matched_index', 'distance')
    ordered_pairs = ordered_pairs[ordered_pairs.distance < distance_cutoff]
    match = match_data.ix[~match_data.gene_name.isin(query_set), :]
    query = match_data.ix[match_data.gene_name.isin(query_set), :]
    query = query.iloc[ordered_pairs.query_index, :]
    match = match.iloc[ordered_pairs.matched_index, :]
    return query, match

def closest_L2_pairs(needles, haystack):
    n_needles = needles.shape[0]
    n_haystack = haystack.shape[0]
    dists = np.zeros((n_needles, n_haystack))
    dists = np.sqrt((dists.T + np.sum(np.square(needles), axis=1)).T + np.sum(np.square(haystack), axis=1) - 2* needles.dot(haystack.T))
    return dists

def studentize(column):
    return (column - np.mean(column))/(np.std(column))

def matched_pairs_multiple(match_data_original, query_set, query_columns, studentize_cols=False):
    match_data = match_data_original.copy()
    if studentize_cols:
        for column in query_columns:
            match_data[column] = studentize(match_data[column])
    x = closest_L2_pairs(match_data.ix[match_data.gene_name.isin(query_set), query_columns].as_matrix(),
                 match_data.ix[~match_data.gene_name.isin(query_set), query_columns].as_matrix(),
                )
    
    mins = x.argmin(axis=1)
    used_elements = set()
    pairs = []
    
    for i in range(x.shape[0]):
        if mins[i] not in used_elements:
            pairs.append((i, mins[i], x[i, mins[i]]))
            used_elements.add(mins[i])
    
    ordered_pairs = pd.DataFrame(pairs)
    ordered_pairs.columns = ('query_index', 'matched_index', 'distance')
    match = match_data.ix[~match_data.gene_name.isin(query_set), :]
    query = match_data.ix[match_data.gene_name.isin(query_set), :]
    query = query.iloc[ordered_pairs.query_index, :]
    match = match.iloc[ordered_pairs.matched_index, :]
    return query, match
