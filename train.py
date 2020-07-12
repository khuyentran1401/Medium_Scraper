import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib
import datapane as dp

from pipeline import pipeline
import hydra
from hydra import utils
import os

from process_data import ClapsToNumerical

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBClassifier, XGBRegressor

from mlflow import log_metric, log_param, log_artifact




@hydra.main(config_path='config.yaml')
def run_training(config):
    """Train the model."""

    #print(os.getcwd())
    current_path = utils.get_original_cwd() + "/"

    # read training data
    data = pd.read_csv(current_path + config.dataset.data)

    transform_claps = ClapsToNumerical()

    transform_claps.fit(data, data[config.target.target])
    data[config.target.target] = transform_claps.transform(data)

    pipe = pipeline(config)
    #X = pipe.fit_transform(data.drop(config.target.target, axis=1),
    #                   data[config.target.target])

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.target.target, axis=1),
        data[config.target.target],
        random_state=1)  # we are setting the seed here
    
    
    #xgb = XGBRegressor(learning_rate=config.xgboost_params.learning_rate,
    #                   colsample_bytree=config.xgboost_params.colsample_bytree,
    #                   min_child_weight=config.xgboost_params.min_child_weight,
    #                   max_depth=config.xgboost_params.max_depth,
    #                   n_estimators=config.xgboost_params.n_estimators)

    pipe.fit(X_train, y_train)
    #pipe.fit(X_train, y_train)

    #columns = pipe.named_steps['get_values'].columns
    #dp.Blob.upload_obj(columns, 'columns')

    #print(X_train.shape)

    #dp.Blob.upload_obj(xgb, name = config.model.model01)
    dp.Blob.upload_obj(pipe, name=config.pipeline.preprocessing)

    pred = pipe.predict(X_test)

    pred = np.round(pred)

    print('pred', pred)
    print('y', y_test)
    # determine mse and rmse
    print('test mse: {}'.format(int(
        mean_squared_error(y_test, np.exp(pred)))))
    print('test rmse: {}'.format(int(
        np.sqrt(mean_squared_error(y_test, np.exp(pred))))))
    print('test r2: {}'.format(
        r2_score(y_test, np.exp(pred))))
    print('classification_report: {}'.format(
        classification_report(y_test, pred)))


if __name__ == '__main__':
    run_training()
