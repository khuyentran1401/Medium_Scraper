from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

import process_data as pc
import hydra

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import AdaBoostClassifier


def pipeline(config):

    nontext = Pipeline(
        [
            ('categorical_imputer',
             pc.CategoricalImputer(['Publication'])),
            ('add_frequency',
             pc.AddFrequency(frequent_variables=config.variables.frequent_features)),
            ('numerical_imputer',
             pc.NumericalImputer(variables=config.variables.impute_numerical_vars)),
            ('find_weekday',
             pc.FindWeekDay()),
            ('drop_features',
             pc.DropFeatures(variables=config.variables.drop_features)),
            ('categorical_to_numerical',
             pc.CategoricalToNumerical()),
        ]
    )
    #tfidf_vectorizer = ColumnTransformer(
    #    transformers=[('vectorizer',
    #                   TfidfVectorizer(use_idf=True, min_df=0),
    #                   ['Title', 'Subtitle'])],
    #    verbose=1
    #)

    subtitle = Pipeline(
        [
            ('categorical_imputer', pc.CategoricalImputer(['Subtitle'])),
            ('preprocess_text', pc.ProcessText(['Subtitle'])),
            ('tfidf_vectorizer', TfidfVectorizer(use_idf=True, min_df=0))
        ],
        verbose=1)

    title = Pipeline(
        [
            ('title_imputer', pc.TitleImputer()),
            ('preprocess_text', pc.ProcessText(['Title'])),
            ('tfidf_vectorizer', TfidfVectorizer(use_idf=True, min_df=0))
        ],
        verbose=1)

    # text = ColumnTransformer(
    #    transformers=[('vectorize_text', text_vectorizer, config.variables.text_features)],
    #    verbose=1
    # )

    preprocess = FeatureUnion(
        [
            ('nontext', nontext),
            ('subtitle', subtitle),
            ('title', title)],
        verbose=1
    )

    pipe = Pipeline(
        [
            ('preprocess', preprocess),
            ('get_values', pc.ProcessMatrix()),
            ('adaboost', AdaBoostClassifier())
            ('xgb_regressor',
             XGBRegressor(learning_rate=config.xgboost_params.learning_rate,
                          colsample_bytree=config.xgboost_params.colsample_bytree,
                          min_child_weight=config.xgboost_params.min_child_weight,
                          max_depth=config.xgboost_params.max_depth,
                          n_estimators=config.xgboost_params.n_estimators))
        ],
        verbose=1)

    return pipe
