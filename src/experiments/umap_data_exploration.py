import os
import sys
import umap
import pandas as pd
import warnings
from pyod.models.iforest import IForest
from general_functions.elliptic_data_preprocessing import load_elliptic_data, setup_train_test_idx, train_test_split
from reaml.models import batch_pyod_per_contamination_level
from general_functions.plotting import plot_UMAP_projection

# Set root directory and add to sys.path
ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

# Supress warnings
warnings.filterwarnings('ignore')

# Define variables
last_train_time_step = 34
last_time_step = 49
only_labeled = True

print("Loading Elliptic dataset...")
X, y = load_elliptic_data(only_labeled=only_labeled)

print(f"Dataset loaded. Shape of X: {X.shape}, Length of y: {len(y)}")

# Setup train/test indices
print(f"Setting up train/test indices with last_train_time_step={last_train_time_step}, last_time_step={last_time_step}")
train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)

# Split data into train and test sets
print("Splitting dataset into train and test sets...")
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)
print(f"Train set size: {X_train_df.shape}, Test set size: {X_test_df.shape}")

# Set contamination levels for predictions
print("Setting contamination levels for prediction.")
contamination_levels = [0.1]
print(f"Contamination levels set: {contamination_levels}")

# Initialize and run models (in this case, Isolation Forest)
print("Running anomaly detection using Isolation Forest (IForest)...")
model_predictions, model_predicted_scores = batch_pyod_per_contamination_level(X_train_df, X_test_df, y_train,
                                                                               contamination_levels, predict_on='test',
                                                                               model_dict={'IF': IForest()})
print("Anomaly detection completed.")

# Define data subsets for UMAP plot
print("Defining data subsets to plot.")
X_subset = X_test_df
y_true_subset = y_test
print(f"Subset size: {X_subset.shape}, Labels count: {len(y_true_subset)}")

# Run UMAP for dimensionality reduction
print("Running UMAP dimensionality reduction...")
embedding = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=70).fit_transform(X_subset)
embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
embedding_df['class'] = y_true_subset.tolist()
embedding_df['class'] = embedding_df['class'].replace({1: 'Illicit', 0: 'Licit'})
print("UMAP embedding completed.")

# Add model predictions to the embedding dataframe
model = 'IF'
contamination_level = 0.1
print(f"Assigning model predictions for {model} at contamination level {contamination_level}.")
embedding_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in model_predictions[model][contamination_level]]

# Plot UMAP projections with predicted labels
print("Plotting UMAP projections with predicted labels.")
plot_UMAP_projection(embedding_df=embedding_df, hue_on='prediction', fontsize=19, labelsize=22,
                     palette=['cadetblue', 'coral'],
                     savefig_path=os.path.join(ROOT_DIR, 'output/figure_3_umap_predicted_label.png'))
print("UMAP plot with predicted labels saved.")

# Plot UMAP projections with true labels
print("Plotting UMAP projections with true labels.")
plot_UMAP_projection(embedding_df=embedding_df, hue_on='class', fontsize=19, labelsize=22,
                     palette=['cadetblue', 'coral'],
                     savefig_path=os.path.join(ROOT_DIR, 'output/figure_4_umap_true_label.png'))
print("UMAP plot with true labels saved.")
