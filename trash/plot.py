def plot_hist_line(df, savename, dpi=200, n_bins=50):

    plt.rcParams.update({'font.size': 50})

    fig, ax = plt.subplots(figsize=(20, 20))

    dff = pd.DataFrame(columns=[df.columns[0], "counts", "bin_centers"])

    for i in np.unique(df[df.columns[0]].loc[df[df.columns[0]] != 0]):
        counts, bin_edges = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == i], bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        new_rows = pd.DataFrame({df.columns[0]: [i] * len(counts), 'counts': counts/np.sum(counts), 'bin_centers': bin_centers})
        dff = new_rows.copy() if dff.empty else pd.concat([dff, new_rows], ignore_index=True)

    sns.lineplot(dff, ax=ax, x="bin_centers", y="counts", hue=df.columns[0], palette=sns.color_palette(), linewidth=5)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel(r'probability')
    ax.set_xlabel(f'{df.columns[1]}')

    ax.set_yscale("log")
    ax.set_xscale("log")

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_dist(df, savename, dpi=200, n_bins=25):

    # REQUIRES DATA FOR THRESH = 0

    plt.rcParams.update({'font.size': 50})

    fig, ax = plt.subplots(figsize=(20, 20))

    dff = pd.DataFrame(columns=[df.columns[0], "counts", "bin_centers"])

    counts_0, bin_edges = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == 0], bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in np.unique(df[df.columns[0]].loc[df[df.columns[0]] != 0]):
        counts, _ = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == i], bins=n_bins)
        counts = counts / counts_0

        new_rows = pd.DataFrame({df.columns[0]: [i]*len(counts), 'counts': counts, 'bin_centers': bin_centers})
        dff = new_rows.copy() if dff.empty else pd.concat([dff, new_rows], ignore_index=True)

    sns.histplot(dff, ax=ax, x="bin_centers", weights="counts", hue=df.columns[0], element="step",
                 palette=sns.color_palette(), bins=n_bins)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel(r'probability')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()