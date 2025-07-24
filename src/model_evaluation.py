import pandas as pd
import altair as alt
from numpy.typing import ArrayLike, NDArray
from typing import Union
from numbers import Number
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

alt.data_transformers.enable('vegafusion')

from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted


def scorer(binary_label: bool=False):
    """Creates scoring function that is used for model evaluation.

    Parameters
    ----------
    binary_label : bool, optional
        True if target label is set as binary, by default False

    Returns
    -------
    sklearn.metrics._scorer._Scorer
        sklearn Scorer object

    Raises
    ------
    TypeError
        If binary_label is not of type bool.
    """
    if not isinstance(binary_label, bool):
        raise TypeError(f"Expected binary_label to be of type bool, instead got: {type(binary_label)}")

    if binary_label:
        return make_scorer(f1_score, pos_label=0)
    
    return make_scorer(f1_score, pos_label='benign')

# Code adapted from DSCI 571: Lecture 4 
def mean_std_cross_val_scores(model, X_train, y_train):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """
    try:
        scores = cross_validate(
            model, X_train, y_train, scoring=make_scorer(f1_score, pos_label='benign'), 
            n_jobs=-1, return_train_score=True
        )
    except:
        scores = cross_validate(
            model, X_train, LabelEncoder().fit_transform(y_train), scoring=make_scorer(f1_score, pos_label=0), 
            n_jobs=-1, return_train_score=True
        )

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def tidy_cv_results(cv_results: dict) -> pd.DataFrame:
    """
    Convert cross-validation results dictionary to a tidy pandas DataFrame.

    Parameters
    ----------
    cv_results : dict
        Dictionary containing cross-validation results with keys 'fit_time',
        'score_time', 'test_score', and 'train_score'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Fit time (s)', 'Score time (s)', 'Train score', 
        and 'Valid score'.

    Raises
    ------
    TypeError
        If cv_results is not a dictionary.
    ValueError
        If cv_results doesn't contain the expected keys.
        
    Examples
    --------
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> cv_results = cross_validate(model, X, y, return_train_score=True)
    >>> tidy_results = tidy_cv_results(cv_results)
    """

    if not isinstance(cv_results, dict):
        raise TypeError(f"Expected dict, instead got {type(cv_results)}")

    df = pd.DataFrame(cv_results)

    expected_index = ['fit_time', 'score_time', 'test_score', 'train_score']
    if df.index.to_list() != expected_index:
        raise ValueError(f"Expected the following index: {expected_index}, received: {df.index.to_list()}")
    
    new_df = pd.DataFrame({
        'Fit time (s)': df.loc['fit_time'],
        'Score time (s)': df.loc['score_time'],
        'Train score': df.loc['train_score'],
        'Valid score': df.loc['test_score']
    })

    return new_df

def get_xgb_feature_importances(model: XGBClassifier, ct: ColumnTransformer) -> pd.DataFrame:
    """
    Extract feature importances from an XGBoost model and match them with feature names from a ColumnTransformer.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost classifier model from which to extract feature importances
    ct : ColumnTransformer
        Column transformer used for preprocessing data before training the model

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importances sorted in descending order with columns:
        - index: feature names (cleaned by removing transformer prefix)
        - weight: importance score for each feature

    Raises
    ------
    ValueError
        If the number of weights from the model doesn't match the number of transformed feature names

    Notes
    -----
    The function cleans feature names by removing the transformer prefix (everything before '__')
    """

    weights = model.feature_importances_
    feature_names = ct.get_feature_names_out()
    feature_names_clean = [name.split("__", 1)[1] for name in feature_names]

    if len(weights) != len(feature_names_clean):
        raise ValueError(f"Dimension mismatch: weights has length {len(weights)}, but features has length {len(feature_names_clean)}")

    feature_importances = pd.DataFrame(
        weights, index=feature_names_clean, columns=['weight']
    ).sort_values(
        'weight', ascending=False
    )
    
    return feature_importances

def generate_feature_importances_chart(df: pd.DataFrame, max_features: int=0, title: str="Feature Importances", width: int=400, height: int=300) -> alt.Chart:
    """
    Generate an Altair chart displaying feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature importances. Expected to have an index of feature names
        and a column named 'weight' for importance values.
    max_features : int, default 0
        Maximum number of features to display. If 0, all features are shown.
        Must be non-negative.
    title : str, default "Feature Importances"
        Title of the chart.
    width : int, default 400
        Width of the chart in pixels.
    height : int, default 300
        Height of the chart in pixels.

    Returns
    -------
    alt.Chart
        An Altair chart object displaying feature importances as a horizontal bar chart.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame or max_features is not an integer.
    ValueError
        If max_features is negative.

    Notes
    -----
    Features are sorted by importance in descending order.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be pandas DataFrame, got {type(df)} instead")
    
    if not isinstance(max_features, int):
        raise TypeError(f"Expected max_features to be int, got {type(max_features)} instead")
    
    if max_features < 0:
        raise ValueError(f"Expected max_features to be >= 0, got {max_features} instead")
        
    if max_features != 0:
        df = df[:max_features]    

    chart = alt.Chart(
        df.reset_index()
    ).mark_bar().encode(
        x=alt.X('weight:Q', title='Feature Importance'),
        y=alt.Y('index:N', sort='-x', title='Feature'),
        tooltip=['index', 'weight']
    ).properties(
        title=title,
        width=width,
        height=height
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    ).configure_title(
        fontSize=14
    )

    return chart

def get_model_metrics(y_true: Union[NDArray, ArrayLike], y_pred: Union[NDArray, ArrayLike]) -> dict:
    """
    Calculate various binary classification metrics based on true and predicted labels.

    Parameters
    ----------
    y_true : Union[NDArray, ArrayLike]
        Array-like object containing the true binary labels.
    y_pred : Union[NDArray, ArrayLike]
        Array-like object containing the predicted binary labels.

    Returns
    -------
    dict
        Dictionary containing various metrics for the 'benign' or 0 class, including:
        - precision
        - recall
        - f1-score
        - support
        - false_malicious_rate : rate of benign samples incorrectly classified as malicious or 1
        - false_benign_rate : rate of malicious samples incorrectly classified as benign or 0

    Raises
    ------
    TypeError
        If the classification report is not returned as a dictionary.
    KeyError
        If 'benign' or 0 is not a key in the classification report.

    Notes
    -----
    This function assumes binary classification with 'benign' or 0 as one of the classes.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    report = classification_report(y_true, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fmr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fbr = fn / (fn + tp) if (fn + tp) > 0 else 0

    if not isinstance(report, dict):
        raise TypeError(f"Expected classification report to be a dictionary, got {type(report)} instead")
        
    target_key = 'benign'
    if target_key not in report and 0 not in report:
        raise KeyError(f"Expected 'benign' or 0 to be a key in the classification report. Available keys: {list(report.keys())}")
    
    target_key = target_key if target_key in report else 0
    
    results = report[target_key]
    results['false_malicious_rate'] = float(fmr)
    results['false_benign_rate'] = float(fbr)

    return results

def generate_tuning_boxplot(
    model: RandomizedSearchCV, 
    height: Union[int, float] = 5, 
    width: Union[int, float] = 2, 
    title: str = "Hyperparameter tuning results"
) -> Figure:
    """
    Generate a boxplot visualization to compare training and validation scores from hyperparameter tuning.

    Parameters
    ----------
    model : RandomizedSearchCV
        A fitted sklearn RandomizedSearchCV object containing the cross-validation results.
    height : Union[int, float], optional
        The height of the figure in inches, by default 5.
    width : Union[int, float], optional
        The width of the figure in inches, by default 2.
    title : str, optional
        The title of the plot, by default "Hyperparameter tuning results".

    Returns
    -------
    Figure
        A matplotlib Figure object containing the boxplot visualization.

    Raises
    ------
    TypeError
        If model is not a RandomizedSearchCV object, height/width are not numbers,
        or title is not a string.
    ValueError
        If height or width are not positive numbers.
    """
    if not isinstance(model, RandomizedSearchCV):
        raise TypeError(f"Expected model to be a scikit-learn RandomizedSearchCV object, got {type(model)} instead")
    
    check_is_fitted(model)

    if not (isinstance(height, Number) and isinstance(width, Number)):
        raise TypeError(f"Expected height, width to both be float, got {type(height)}, {type(width)} instead")
    
    if not (height > 0 and width > 0):
        raise ValueError(f"Expected height, width to be positive numbers, got {height}, {width} instead")

    if not isinstance(title, str):
        raise TypeError(f"Expected title to be str, got {type(title)} instead")
    
    fig = plt.figure(figsize=(5, 2), dpi=300)

    data = [model.cv_results_['mean_test_score'], model.cv_results_['mean_train_score']]
    labels = ['Valid', 'Train']

    plt.boxplot(
        data, tick_labels=labels, patch_artist=True, orientation='horizontal',
        widths=0.5
    )

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.title('Hyperparameter tuning results', fontsize=14)
    plt.xlabel('F1 Score')
    plt.xlim(0.6, 1.0)
    
    return fig
