def pca_dashboard(original_data, pca_data, pca, hue):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    fig = plt.figure(figsize=(12,10))
    fig.suptitle("Principal Component Analysis")
    
    ax_2D = fig.add_subplot(2,2,1)

    sns.scatterplot(ax=ax_2D, data=pca_data, x="PC1", y="PC2", hue=hue, alpha=0.5)

    ax_3D = fig.add_subplot(2,2,2, projection='3d')

    scatter_x = pca_data[["PC1"]].values
    scatter_y = pca_data[["PC2"]].values
    scatter_z = pca_data[["PC3"]].values
    hue_group = pca_data[[hue]].values
    
    ax_3D.scatter(scatter_x, scatter_y, scatter_z, c=hue_group)

    ax_3D.set_xlabel("PC1")
    ax_3D.set_ylabel("PC2")
    ax_3D.set_zlabel("PC3")
    ax_3D.dist=9
    # fig.colorbar(points)

    # explained_variance_ratio = pd.DataFrame(data=pca.explained_variance_ratio_, index=["PC1", "PC2", "PC3"])

    # ax_evr = fig.add_subplot(2,2,3)
    # sns.barplot(ax=ax_evr, data=explained_variance_ratio.transpose())
    # ax_evr.set(
    #     ylabel="Explained Variance Ratio"
    # )

    # loading_score_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)

    # loading_scores = pd.DataFrame(data=loading_score_matrix, columns=["PC1", "PC2", "PC3"], index=original_data.columns)

    # most_influential = pd.concat([loading_scores['PC1'].sort_values(axis='rows').head(5), loading_scores['PC1'].sort_values(axis='rows').tail(5)], axis='index')

    # ax_ls = fig.add_subplot(2,2,4)
    # most_influential.plot.barh(ax=ax_ls)
    # ax_ls.set(
    #     xlabel="Loading Scores"
    # )

    plt.tight_layout()