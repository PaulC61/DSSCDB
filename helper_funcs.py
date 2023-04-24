from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def pca_dashboard(original_data, pca_data, pca, hue, continuous=False, title="Default"):
    
    fig = plt.figure(figsize=(12,10))
    fig.suptitle(f"{title} Principal Component Analysis")
    
    ax_2D = fig.add_subplot(2,2,1)

    sns.scatterplot(ax=ax_2D, data=pca_data, x="PC1", y="PC2", hue=hue, alpha=0.5)

    ax_3D = fig.add_subplot(2,2,2, projection='3d')

    scatter_x = pca_data[["PC1"]].values
    scatter_y = pca_data[["PC2"]].values
    scatter_z = pca_data[["PC3"]].values
    hue_group = pca_data[[hue]].values
    
    if continuous:
        ax_3D.scatter(scatter_x, scatter_y, scatter_z, c=hue_group)
    else:
        for g in np.unique(hue_group):
            i = np.where(hue_group==g)
            ax_3D.scatter(scatter_x[i], scatter_y[i], scatter_z[i], label=g)

    ax_3D.set_xlabel("PC1")
    ax_3D.set_ylabel("PC2")
    ax_3D.set_zlabel("PC3")
    ax_3D.dist=9

    explained_variance_ratio = pd.DataFrame(data=pca.explained_variance_ratio_, index=["PC1", "PC2", "PC3"])

    ax_evr = fig.add_subplot(2,2,3)
    sns.barplot(ax=ax_evr, data=explained_variance_ratio.transpose())
    ax_evr.set(
        ylabel="Explained Variance Ratio"
    )

    loading_score_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)

    loading_scores = pd.DataFrame(data=loading_score_matrix, columns=["PC1", "PC2", "PC3"], index=original_data.columns)

    most_influential = pd.concat([loading_scores['PC1'].sort_values(axis='rows').head(5), loading_scores['PC1'].sort_values(axis='rows').tail(5)], axis='index')

    ax_ls = fig.add_subplot(2,2,4)
    most_influential.plot.barh(ax=ax_ls)
    ax_ls.set(
        xlabel="Loading Scores"
    )

    plt.tight_layout()


def bic_score(X, labels):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is directly translated from the GoLang code made by the author of the paper. 
    The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """

    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name].to_numpy()
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += \
            n_points_cluster * np.log(n_points_cluster) \
            - n_points_cluster * np.log(n_points) \
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
            - (n_points_cluster - 1) / 2

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
        
    return bic


def regression_results(reg_model, X_train, X_test, Y_train, Y_test, title="Default:"):

    pred_train = reg_model.predict(X_train)
    pred_test = reg_model.predict(X_test)
    train_df = pd.DataFrame.from_dict({'True': Y_train, 'Predicted': pred_train})
    test_df = pd.DataFrame.from_dict({'True': Y_test, 'Predicted': pred_test})


    # ---- train results ----
    r2 = r2_score(Y_train, pred_train)
    train_mae = mean_absolute_error(Y_train, pred_train)

    # ---- test results ----
    q2 = r2_score(Y_test, pred_test)
    test_mae = mean_absolute_error(Y_test, pred_test)


    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    fig.suptitle(f'{title} RandomForest Regression Model')

    # Training Set Results
    sns.scatterplot(ax=axes[0], data=train_df, x='True', y='Predicted')
    axes[0].set_title(f"Training Set: r2 -> {round(r2,2)}, mae -> {round(train_mae, 2)}")

    # Test Set Results
    sns.scatterplot(ax=axes[1], data=test_df, x='True', y='Predicted')
    axes[1].set_title(f"Test Set: q2 -> {round(q2, 2)}, mae -> {round(test_mae, 2)}")


def gmm_bic_search(descriptors):
    
    n_range = range(2,11)

    bic_score = []
    aic_score = []

    for n in n_range:
        gm = GaussianMixture(n_components=n,
                            random_state=42,
                            n_init=10
                            )
        gm.fit(descriptors)
        bic_score.append(gm.bic(descriptors))

    gm_df = pd.DataFrame.from_dict({"n_components": n_range, "BIC": bic_score})

    sns.lineplot(data=gm_df, x="n_components", y="BIC")