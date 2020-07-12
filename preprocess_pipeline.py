from sklearn.pipeline import Pipeline
import process_data as pc
import hydra

from xgboost import XGBClassifier



def pipeline(config):

    preprocessed_pipe = Pipeline(

        [
            ('title_imputer',
             pc.TitleImputer()),
            ('claps_to_numerical',
             pc.ClapsToNumerical()),
            ('add_frequency',
             pc.AddFrequency(frequent_variables=config.variables.frequent_features)),
            ('numerical_imputer',
             pc.NumericalImputer(variables=config.variables.impute_numerical_vars)),
            ('find_weekday',
             pc.FindWeekDay()),
            ('drop_features',
             pc.DropFeatures(variables=config.variables.drop_features)),
            ('categorical_imputer',
             pc.CategoricalImputer(variables=config.variables.impute_categorical_vars)),
            ('categorical_to_numerical',
             pc.CategoricalToNumerical()),
            ('df_to_sparse',
             pc.DfToSparse(variables=confid.variables.text_features, target=config.target.target)),
            
        ]
    )
