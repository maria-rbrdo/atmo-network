def PCC(data, lag):
    """Find linear Pearson correlation matrix from data with (lag = True) or without
     (lag = False) lag."""

    if lag == 0:
        return np.corrcoef(data)
    else:
        cm = np.zeros(shape=(len(data), len(data)))
        np.fill_diagonal(cm, 1)
        means = np.mean(data, 1)
        stds = np.std(data, 1)
        with (alive_bar(manual=True, force_tty=True) as bar):
            for i in range(len(data)):
                x_i = data[i, :]
                for j in range(i + 1, len(data)):
                    x_j = data[j, :]
                    corr = np.correlate(x_i-means[i], x_j-means[j], mode='same')/len(x_i)/(stds[i]*stds[j])
                    max_corr = np.max(corr)
                    cm[i, j] = max_corr
                    cm[j, i] = max_corr
                bar(np.count_nonzero(cm)/len(cm)**2)
        return cm