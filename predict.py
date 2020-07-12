import pandas as pd

import joblib

from sklearn.metrics import classification_report
import hydra

# test pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hydra import utils

from pipeline import pipeline

import datapane as dp

from process_data import ClapsToNumerical

import subprocess


def make_prediction(input_test, config):

    #_pipe_match = joblib.load(
    #    filename=utils.to_absolute_path(config.pipeline.pipeline01))

    model = dp.Blob.get(name=config.model.model01).download_obj()

    results = model.predict(input_test)

    return results


@hydra.main(config_path='config.yaml')
def predict(config):

    current_path = utils.get_original_cwd() + "/"

    X_test = pd.read_csv(current_path + config.dataset.test)

    preprocessing_pipe = dp.Blob.get(name = config.pipeline.preprocessing).download_obj()
    X_test = preprocessing_pipe.predict(X_test)

    
    print(X_test)
    pred = make_prediction(X_test, config)

    pred = np.round(pred)

    # determine mse and rmse
    print('test mse: {}'.format(int(
        mean_squared_error(y_test, np.exp(pred)))))
    print('test rmse: {}'.format(int(
        np.sqrt(mean_squared_error(y_test, np.exp(pred))))))
    print('test r2: {}'.format(
        r2_score(y_test, np.exp(pred))))
    print('classification_report: {}'.format(
        classification_report(y_test, pred)))

if __name__=='__main__':
        

    test = pd.DataFrame(data=np.array([['A/B Testing: The Basics!',
                                        'What, Why, When and How',
                                        1,
                                        'Divya Naidu',
                                        'Towards Data Science',
                                        2020,
                                        7,
                                        5,
                                        'data_science',
                                        7,
                                        None,
                                        None]]),
                        columns=['Title',
                                 'Subtitle',
                                 'Image',
                                 'Author',
                                 'Publication',
                                 'Year',
                                 'Month',
                                 'Day',
                                 'Tag',
                                 'Reading_Time',
                                 'url',
                                 'Author_url'])

    test.to_csv('test.csv', index=False)
    predict()
