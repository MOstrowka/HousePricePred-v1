import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

def scatter_plot(df, x_col, y_col, figsize=(10, 6)):
    """
    Creates a scatter plot for two columns from a DataFrame.

    :param df: DataFrame containing the data
    :param x_col: Name of the column for the X axis
    :param y_col: Name of the column for the Y axis
    """
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

def box_plot(df, x_col, y_col, figsize=(10, 6), rotate_xticks=False):
    """
    Creates a box plot for a categorical and a numerical column from a DataFrame using Seaborn.

    :param df: DataFrame containing the data
    :param x_col: Name of the categorical column for the X axis
    :param y_col: Name of the numerical column for the Y axis
    :param figsize: Tuple representing the figure size (width, height)
    :param rotate_xticks: Boolean to rotate x-axis labels for better readability
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(f'{y_col} by {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    if rotate_xticks:
        plt.xticks(rotation=90)
    
    plt.show()

def evaluate_model(model, X_val, y_val, scaler_y, plot=True):
    """
    Evaluates a regression model on the validation set using various metrics.

    :param model: The trained regression model to evaluate
    :param X_val: Validation features
    :param y_val: Validation labels
    :param scaler_y: Scaler used to inverse transform predictions to original scale
    :param plot: Boolean to control whether to display plots
    :return: Dictionary containing evaluation metrics
    """
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Inverse transform predictions to original scale
    y_val_pred_original = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    # Calculate evaluation metrics
    metrics = {
        "Validation RMSE": np.sqrt(mean_squared_error(y_val_original, y_val_pred_original)),
        "Validation MAE": mean_absolute_error(y_val_original, y_val_pred_original),
        "Validation R2": r2_score(y_val_original, y_val_pred_original)
    }
    
    # Print evaluation metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot actual vs predicted values
    if plot:
        plt.figure(figsize=(10, 5))
        plt.scatter(y_val_original, y_val_pred_original, alpha=0.2)
        plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'r--', lw=2)
        plt.xlabel('True price')
        plt.ylabel('Predicted price')
        plt.title('True vs Predicted price')
        plt.tight_layout()
        plt.show()

    return metrics