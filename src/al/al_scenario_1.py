import pandas as pd

from functions.config import features_path, classes_path, edges_path
from functions.elliptic_dataset_preprocessing import preprocessing_pipeline
from functions.al_experiments import al_experiment
from functions.al_policies import AnomalyDetectionQuery, EllipticEnvelopeQuery, QueryByCommittee, ExpectedModelChange, UncertaintySampling
from functions.custom_logistic_regression import Custom_LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from pyod.models.lof import LOF
from pyod.models.iforest import IForest


warmup_learner_dict = {
    'IF' : AnomalyDetectionQuery(IForest()),
    'LOF' : AnomalyDetectionQuery(LOF()),
    'Elliptic Envelope' : EllipticEnvelopeQuery(EllipticEnvelope(support_fraction = 1.0))
}

hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(RandomForestClassifier())
}

supervised_classifier = {
    'Random Forest' : RandomForestClassifier()
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True, edges_path=edges_path, classes_path=classes_path, features_path=features_path)
al_scenario1_rf = al_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict,
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 60, 
                   batch_size=50, total_budget = 60)


hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(XGBClassifier())
}

supervised_classifiers = {
    'XGBoost' : XGBClassifier()
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True)
al_scenario1_xgb = al_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict,
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 60, 
                   batch_size=50, total_budget = 60)


hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(LogisticRegression())
}

supervised_classifier = {
    'Logistic Regression' : LogisticRegression(max_iter=10000)
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True, edges_path=edges_path, features_path=features_path, classes_path=classes_path)
al_scenario1_lr = al_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict,
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 60, 
                   batch_size=50, total_budget = 60)



combined_df = pd.concat([al_scenario1_rf, al_scenario1_xgb, al_scenario1_lr], ignore_index=True)
df_pivot = combined_df.pivot_table(index=['warm-up learner', 'hot-learner', 'supervised classifier'], columns='labeled pool size', values=['mean', 'std'])