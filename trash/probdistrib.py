# ----------------------------------------------------------------------------------------------------------------------
# Probability distribution:
# ----------------------------------------------------------------------------------------------------------------------
def calc_cum_prob_distrib(am, measure, savename, dpi=200):
    values, base = np.histogram(am, bins=40)  # evaluate histogram
    cumulative = 1 - np.cumsum(values) / len(am)  # evaluate cumulative
    p = np.polyfit(base[:-1], np.log(cumulative), 1, w=np.sqrt(cumulative))  # fit exponential
    approx = np.exp(p[1]) * np.exp(p[0] * base[:-1])

    # make figure
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    df = pd.DataFrame({'x': base[:-1], 'y': cumulative, 'y_fit': approx})
    sns.lineplot(df, x='x', y='y_fit', ax=ax, markersize=5, linewidth=2, color="gray")
    sns.lineplot(df, x='x', y='y', ax=ax, marker='o', markersize=5, linewidth=2, color="black")
    ax.lines[0].set_linestyle("--")

    ax.set_ylim([1e-3, 1])
    ax.set_yscale('log')

    ax.set_xlabel(f'{measure}')
    ax.set_ylabel('cumulative probability distribution')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    fig.clear()

    return cumulative

