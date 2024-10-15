import seaborn as sns
import matplotlib.pyplot as plt


def plot_UMAP_projection(embedding_df, hue_on='class', labelsize=20, fontsize=22, palette=['cadetblue', 'coral'],
                         linewidth=0.000001, savefig_path=None):
    fig_dims = (10, 6)
    fig, ax = plt.subplots(figsize=fig_dims)

    # Plot the UMAP projection using Seaborn scatterplot
    sns.scatterplot(x='dim_0', y='dim_1', hue=hue_on, style=hue_on, markers=['.', 'X'], size=hue_on,
                    sizes=[150, 170], linewidth=linewidth, palette=palette, data=embedding_df, ax=ax)

    # Set axis labels
    ax.set_xlabel('Dimension 1', fontsize=fontsize)
    ax.set_ylabel('Dimension 2', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')

    # Customize the legend
    L = ax.legend(prop={'size': labelsize}, facecolor="#EEEEEE", handletextpad=-0.5)

    # Check if the legend contains the expected number of items
    legend_texts = L.get_texts()

    if len(legend_texts) > 0:
        if hue_on == 'class':
            legend_texts[0].set_text('True label')  # Modify the title of the legend
        else:
            legend_texts[0].set_text('Predicted label')

        if len(legend_texts) > 1:
            legend_texts[1].set_text("licit")  # Modify first class name

        if len(legend_texts) > 2:
            legend_texts[2].set_text("illicit")  # Modify second class name

    # Display or save the plot
    if savefig_path is None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)

    fig.show()
