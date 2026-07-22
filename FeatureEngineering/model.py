from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import networkx as nx
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from config import *
from FeatureEngineering.data import compute_seed_features

def train_random_forest(
    df,
    target_column="match_ratio",
    test_size=0.2,
    random_state=42,
    n_estimators=200,
):
    """
    Trains a Random Forest regressor on the given dataframe.

    Returns
    -------
    model : RandomForestRegressor
    metrics : dict
    feature_importance : pandas.Series
    predictions : numpy.ndarray
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    model.fit(X_train, y_train)


    predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),

    }

    feature_importance = (
        pd.Series(
            model.feature_importances_,
            index=X.columns,
        )
        .sort_values(ascending=False)
    )

    return (
        model,
        metrics,
        feature_importance,
        predictions,
    )

def predict_seed_score(model, G1_nx, G2_nx, seeds_G1, seeds_G2):
    """
    Predicts the graph matching accuracy of a seed set using the
    trained Random Forest model. ONLY CALL IF RF_MODEL IS INITIALIZED
    """
    features = compute_seed_features(
        G1_nx,
        G2_nx,
        seeds_G1,
        seeds_G2
    )

    features["num_seeds"] = len(seeds_G1)

    X = pd.DataFrame([features])

    return model.predict(X)[0]