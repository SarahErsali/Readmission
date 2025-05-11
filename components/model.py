import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTETomek
from bayes_opt import BayesianOptimization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import joblib


################### Loading, Cleaning and Spliting Dataset ##################

def load_and_clean_data(file_path='components/data/cleaned_dataset_no_ghosts.csv'):
    # Load data
    c_data = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    c_data = c_data.drop(columns=[
        'race', 'discharge_disposition_id', 'admission_source_id', 
        'number_outpatient', 'number_emergency', 'number_inpatient'
    ])
    
    # Convert all values to numeric, set non-convertible to NaN
    #c_data = c_data.apply(pd.to_numeric, errors='coerce')
    
    # Merge classes: set readmitted = 1 if not 0
    c_data['readmitted'] = np.where(c_data['readmitted'] == 0, 0, 1)
    
    # Features and target
    X = c_data.drop(columns=['readmitted'])
    y = c_data['readmitted']
    
    #First split: Train (60%) / Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # Second split: Validation (20%) / Test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,          # half of the 40% → 20% each
        random_state=42,
        stratify=y_temp
    )
    
        
    return X_train, X_val, X_test, y_train, y_val, y_test


################### Default models ##################


# Default XGBoost model
def train_xgb_default(X_resampled, y_resampled, X_val):
    xgb_default = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=2,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    xgb_default.fit(X_resampled, y_resampled)      
    y_pred_xgb_default = xgb_default.predict(X_val)           
    return xgb_default, y_pred_xgb_default


# Default LightGBM model
def train_lgb_default(X_resampled, y_resampled, X_val):
    lgb_default = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=2,
        random_state=42
    )
    lgb_default.fit(X_resampled, y_resampled)
    y_pred_lgb_default = lgb_default.predict(X_val)
    return lgb_default, y_pred_lgb_default



# Default Random Forest model
def train_rf_default(X_resampled, y_resampled, X_val):
    rf_default = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_default.fit(X_resampled, y_resampled)
    y_pred_rf_default = rf_default.predict(X_val)
    return rf_default, y_pred_rf_default



# Default ensemble using VotingClassifier
def train_ensembeled_default(rf_default, xgb_default, X_resampled, y_resampled, X_val):
    rf_xgb_default = VotingClassifier(
        estimators=[('rf', rf_default), ('xgb', xgb_default)],
        voting='soft'
    )
    rf_xgb_default.fit(X_resampled, y_resampled)
    y_pred_ensemble_default = rf_xgb_default.predict(X_val)
    return rf_xgb_default, y_pred_ensemble_default





################## Tuned models ##################

# Apply class weights for improvment
# Compute class weights for the rebalanced data
def compute_sample_weights(y_smote):
    """
    Compute sample weights based on class distribution for rebalanced data.
    
    Args:
        y_resampled (array-like): Target labels after resampling.
        
    Returns:
        sample_weights (np.array): Array of weights for each sample.
        weight_dict (dict): Class-to-weight mapping.
    """
    classes = np.unique(y_smote)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_smote)
    weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([weight_dict[label] for label in y_smote])  # Create sample weights for each training instance
    return sample_weights, weight_dict




def train_xgb_tuned(X_smote, y_smote, X_test, sample_weights):
    """
    Tunes and trains an XGBoost model using Bayesian Optimization and sample weights.
    
    Returns:
        model: Trained XGBoost model
        y_pred: Predictions on X_test
        best_params: Best hyperparameters found
    """
    
    # Define evaluation function for Bayesian Optimization
    def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )
        scores = cross_val_score(
            model,
            X_smote,
            y_smote,
            cv=3,
            scoring='f1_macro',
            fit_params={'sample_weight': sample_weights}
        )
        return scores.mean()

    # Define hyperparameter search space
    xgb_params = {
        'max_depth': (3, 5),
        'learning_rate': (0.005, 0.01),
        'n_estimators': (50, 700),
        'subsample': (0.5, 0.7),
        'colsample_bytree': (0.6, 0.8)
    }

    # Run Bayesian Optimization
    xgb_bo = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=xgb_params,
        random_state=42
    )
    xgb_bo.maximize(init_points=5, n_iter=20)

    # Extract best parameters
    best_params_xgb = xgb_bo.max['params']
    best_params_xgb['max_depth'] = int(best_params_xgb['max_depth'])
    best_params_xgb['n_estimators'] = int(best_params_xgb['n_estimators'])

    # Train final model with best params
    xgb_tuned = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        **best_params_xgb
    )
    xgb_tuned.fit(X_smote, y_smote, sample_weight=sample_weights)
    y_pred_xgb_tuned = xgb_tuned.predict(X_test)

    return xgb_tuned, y_pred_xgb_tuned, best_params_xgb

#----------------------------------

def train_lgb_tuned(X_smote, y_smote, X_test, sample_weights):
    """
    Tunes and trains a LightGBM model using Bayesian Optimization and sample weights.

    Returns:
        model: Trained LightGBM model
        y_pred: Predictions on X_test
        best_params: Best hyperparameters found
    """
    
    # Define evaluation function
    def lgb_evaluate(max_depth, learning_rate, n_estimators, num_leaves, subsample):
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            num_leaves=int(num_leaves),
            subsample=subsample
        )
        scores = cross_val_score(
            model,
            X_smote,
            y_smote,
            cv=3,
            scoring='f1_macro',
            fit_params={'sample_weight': sample_weights}
        )
        return scores.mean()

    # Define search space
    lgb_params = {
        'max_depth': (3, 6),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 500),
        'num_leaves': (20, 50),
        'subsample': (0.6, 1.0)
    }

    # Run Bayesian Optimization
    lgb_bo = BayesianOptimization(
        f=lgb_evaluate,
        pbounds=lgb_params,
        random_state=42
    )
    lgb_bo.maximize(init_points=5, n_iter=20)

    # Extract best parameters
    best_params_lgb = lgb_bo.max['params']
    best_params_lgb['max_depth'] = int(best_params_lgb['max_depth'])
    best_params_lgb['n_estimators'] = int(best_params_lgb['n_estimators'])
    best_params_lgb['num_leaves'] = int(best_params_lgb['num_leaves'])

    # Train final model
    lgb_tuned = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        **best_params_lgb
    )
    lgb_tuned.fit(X_smote, y_smote, sample_weight=sample_weights)
    y_pred_lgb_tuned = lgb_tuned.predict(X_test)

    return lgb_tuned, y_pred_lgb_tuned, best_params_lgb

#--------------------------------

def train_rf_tuned(X_smote, y_smote, X_test, sample_weights):
    """
    Tunes and trains a Random Forest model using Bayesian Optimization and sample weights.

    Returns:
        model: Trained Random Forest model
        y_pred: Predictions on X_test
        best_params: Best hyperparameters found
    """
    
    # Evaluation function for Bayesian Optimization
    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=42,
            class_weight='balanced'
        )
        scores = cross_val_score(
            model,
            X_smote,
            y_smote,
            cv=3,
            scoring='f1_macro',
            fit_params={'sample_weight': sample_weights}
        )
        return scores.mean()

    # Define hyperparameter space
    rf_params = {
        'n_estimators': (50, 300),
        'max_depth': (5, 10),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5),
        'max_features': (0.3, 1.0)
    }

    # Run Bayesian Optimization
    rf_bo = BayesianOptimization(
        f=rf_evaluate,
        pbounds=rf_params,
        random_state=42
    )
    rf_bo.maximize(init_points=5, n_iter=20)

    # Get best params
    best_params_rf = rf_bo.max['params']
    best_params_rf['n_estimators'] = int(best_params_rf['n_estimators'])
    best_params_rf['max_depth'] = int(best_params_rf['max_depth'])
    best_params_rf['min_samples_split'] = int(best_params_rf['min_samples_split'])
    best_params_rf['min_samples_leaf'] = int(best_params_rf['min_samples_leaf'])

    # Train final model
    rf_tuned = RandomForestClassifier(
        n_estimators=best_params_rf['n_estimators'],
        max_depth=best_params_rf['max_depth'],
        min_samples_split=best_params_rf['min_samples_split'],
        min_samples_leaf=best_params_rf['min_samples_leaf'],
        max_features=best_params_rf['max_features'],
        random_state=42,
        class_weight='balanced'
    )
    rf_tuned.fit(X_smote, y_smote, sample_weight=sample_weights)
    y_pred_rf_tuned = rf_tuned.predict(X_test)

    return rf_tuned, y_pred_rf_tuned, best_params_rf


#--------------------------------
def train_ensemble_tuned(rf_tuned, xgb_tuned, X_smote, y_smote, X_test):
    """
    Tunes and trains a VotingClassifier (soft voting) using Bayesian Optimization.

    Args:
        rf_model: Pre-trained Random Forest model
        xgb_model: Pre-trained XGBoost model
        X_smote: Resampled training features
        X_test: Test set features

    Returns:
        ensemble_model: Trained VotingClassifier with best weights
        y_pred: Predictions on X_test
        best_params: Best weights found
    """
    
    # Evaluation function for Bayesian Optimization
    def ensemble_evaluate(w_rf, w_xgb):
        ensemble = VotingClassifier(
            estimators=[('rf', rf_tuned), ('xgb', xgb_tuned)],
            voting='soft',
            weights=[w_rf, w_xgb]
        )
        scores = cross_val_score(
            ensemble,
            X_smote,
            y_smote,
            cv=3,
            scoring='f1_macro'
        )
        return scores.mean()

    # Define parameter bounds
    pbounds = {
        'w_rf': (1, 3),
        'w_xgb': (1, 3)
    }

    # Run Bayesian Optimization
    optimizer = BayesianOptimization(
        f=ensemble_evaluate,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=20)

    # Retrieve best weights
    best_params_ensemble = optimizer.max['params']

    # Train final ensemble
    ensemble_tuned = VotingClassifier(
        estimators=[('rf', rf_tuned), ('xgb', xgb_tuned)],
        voting='soft',
        weights=[best_params_ensemble['w_rf'], best_params_ensemble['w_xgb']]
    )
    ensemble_tuned.fit(X_smote, y_smote)
    y_pred_ensemble_tuned = ensemble_tuned.predict(X_test)

    return ensemble_tuned, y_pred_ensemble_tuned, best_params_ensemble




################### ROC curve ##################

# generic X_eval and y_eval can pass in either val or test set in pipeline

def compute_auc(model, X_eval, y_eval):
    """
    Returns AUC score for a given model.
    """
    y_probs = model.predict_proba(X_eval)[:, 1]

    return roc_auc_score(y_eval, y_probs)



def compute_roc_curve(model, X_eval, y_eval):
    """
    Returns FPR and TPR for ROC curve.
    """
    y_probs = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, y_probs)

    return fpr, tpr




################### KPIs ##################

def kpi_reports(model, X_eval, y_eval):
    """
    Computes classification metrics for a given model.
    Returns a dictionary of accuracy, precision, recall, and f1 scores.
    """
    y_pred = model.predict(X_eval)
    report = classification_report(y_eval, y_pred, output_dict=True)

    result = {
        # Overall
        "Accuracy": accuracy_score(y_eval, y_pred),

        # Only for class 1
        "Precision (class 1)": report["1"]["precision"],
        "Recall (class 1)": report["1"]["recall"],
        "F1-score (class 1)": report["1"]["f1-score"],

        # Macro average
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],

        # Weighted average
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    }

    return result





################### Model Pipeline ##################


def model_pipeline():

    # 1) Load data and get raw splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_clean_data()

    # 2) resample with SMOTETomek
    sm = SMOTETomek()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # 3) Train default models
    xgb_default, y_pred_xgb_default = train_xgb_default(X_resampled, y_resampled, X_val)
    lgb_default, y_pred_lgb_default = train_lgb_default(X_resampled, y_resampled, X_val)
    rf_default, y_pred_rf_default = train_rf_default(X_resampled, y_resampled, X_val)
    rf_xgb_default, y_pred_ensemble_default = train_ensembeled_default(rf_default, xgb_default, X_resampled, y_resampled, X_val)


    # 4) Combined train and val for tuned models
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    # 5) Apply second SMOTETomek on combined data
    sm2 = SMOTETomek()
    X_smote, y_smote = sm2.fit_resample(X_combined, y_combined)

    # 6) Compute sample weights for balancing unbalanced data
    sample_weights, weight_dict = compute_sample_weights(y_smote)

    # 7) Train tuned models
    xgb_tuned, y_pred_xgb_tuned, best_params_xgb = train_xgb_tuned(X_smote, y_smote, X_test, sample_weights)

    lgb_tuned, y_pred_lgb_tuned, best_params_lgb = train_lgb_tuned(X_smote, y_smote, X_test, sample_weights)

    rf_tuned, y_pred_rf_tuned, best_params_rf = train_rf_tuned(X_smote, y_smote, X_test, sample_weights)

    ensemble_tuned, y_pred_ensemble_tuned, best_params_ensemble = train_ensemble_tuned(rf_tuned, xgb_tuned, X_smote, y_smote, X_test)


    # 8) Compute AUC and ROC curve for all models
    auc_xgb_default = compute_auc(xgb_default, X_val, y_val)
    auc_lgb_default = compute_auc(lgb_default, X_val, y_val)
    auc_rf_default = compute_auc(rf_default, X_val, y_val)
    auc_ensemble_default = compute_auc(rf_xgb_default, X_val, y_val)

    auc_xgb_tuned = compute_auc(xgb_tuned, X_test, y_test)
    auc_lgb_tuned = compute_auc(lgb_tuned, X_test, y_test)
    auc_rf_tuned = compute_auc(rf_tuned, X_test, y_test)
    auc_ensemble_tuned = compute_auc(ensemble_tuned, X_test, y_test)

    fpr_xgb_default, tpr_xgb_default = compute_roc_curve(xgb_default, X_val, y_val)
    fpr_lgb_default, tpr_lgb_default = compute_roc_curve(lgb_default, X_val, y_val)
    fpr_rf_default, tpr_rf_default = compute_roc_curve(rf_default, X_val, y_val)
    fpr_ensemble_default, tpr_ensemble_default = compute_roc_curve(rf_xgb_default, X_val, y_val)

    fpr_xgb_tuned, tpr_xgb_tuned = compute_roc_curve(xgb_tuned, X_test, y_test)
    fpr_lgb_tuned, tpr_lgb_tuned = compute_roc_curve(lgb_tuned, X_test, y_test)
    fpr_rf_tuned, tpr_rf_tuned = compute_roc_curve(rf_tuned, X_test, y_test)
    fpr_ensemble_tuned, tpr_ensemble_tuned = compute_roc_curve(ensemble_tuned, X_test, y_test)

    # 9) Create a dictionary for ROC AUC results
    roc_auc_results = {}

    roc_auc_results["XGB Default"] = {
        "auc": compute_auc(xgb_default, X_val, y_val),
        "fpr": compute_roc_curve(xgb_default, X_val, y_val)[0],
        "tpr": compute_roc_curve(xgb_default, X_val, y_val)[1]
    }

    roc_auc_results["LGB Default"] = {
        "auc": compute_auc(lgb_default, X_val, y_val),
        "fpr": compute_roc_curve(lgb_default, X_val, y_val)[0],
        "tpr": compute_roc_curve(lgb_default, X_val, y_val)[1]
    }

    roc_auc_results["RF Default"] = {
        "auc": compute_auc(rf_default, X_val, y_val),
        "fpr": compute_roc_curve(rf_default, X_val, y_val)[0],
        "tpr": compute_roc_curve(rf_default, X_val, y_val)[1]
    }

    roc_auc_results["Ensemble Default"] = {
        "auc": compute_auc(rf_xgb_default, X_val, y_val),
        "fpr": compute_roc_curve(rf_xgb_default, X_val, y_val)[0],
        "tpr": compute_roc_curve(rf_xgb_default, X_val, y_val)[1]
    }

    roc_auc_results["XGB Tuned"] = {
        "auc": compute_auc(xgb_tuned, X_test, y_test),
        "fpr": compute_roc_curve(xgb_tuned, X_test, y_test)[0],
        "tpr": compute_roc_curve(xgb_tuned, X_test, y_test)[1]
    }

    roc_auc_results["LGB Tuned"] = {
        "auc": compute_auc(lgb_tuned, X_test, y_test),
        "fpr": compute_roc_curve(lgb_tuned, X_test, y_test)[0],
        "tpr": compute_roc_curve(lgb_tuned, X_test, y_test)[1]
    }

    roc_auc_results["RF Tuned"] = {
        "auc": compute_auc(rf_tuned, X_test, y_test),
        "fpr": compute_roc_curve(rf_tuned, X_test, y_test)[0],
        "tpr": compute_roc_curve(rf_tuned, X_test, y_test)[1]
    }

    roc_auc_results["Ensemble Tuned"] = {
        "auc": compute_auc(ensemble_tuned, X_test, y_test),
        "fpr": compute_roc_curve(ensemble_tuned, X_test, y_test)[0],
        "tpr": compute_roc_curve(ensemble_tuned, X_test, y_test)[1]
    }


    # 10) Create a dictionary for KPIs
    kpis_dict = {}

    kpis_dict["XGB Default"] = kpi_reports(xgb_default, X_val, y_val)
    kpis_dict["LGB Default"] = kpi_reports(lgb_default, X_val, y_val)
    kpis_dict["RF Default"] = kpi_reports(rf_default, X_val, y_val)
    kpis_dict["Ensemble Default"] = kpi_reports(rf_xgb_default, X_val, y_val)

    kpis_dict["XGB Tuned"] = kpi_reports(xgb_tuned, X_test, y_test)
    kpis_dict["LGB Tuned"] = kpi_reports(lgb_tuned, X_test, y_test)
    kpis_dict["RF Tuned"] = kpi_reports(rf_tuned, X_test, y_test)
    kpis_dict["Ensemble Tuned"] = kpi_reports(ensemble_tuned, X_test, y_test)


    # 11) Create a dictionary for models
    models = {
        "XGB Default": xgb_default,
        "LGB Default": lgb_default,
        "RF Default": rf_default,
        "Ensemble Default": rf_xgb_default,
        "XGB Tuned": xgb_tuned,
        "LGB Tuned": lgb_tuned,
        "RF Tuned": rf_tuned,
        "Ensemble Tuned": ensemble_tuned,
    }


    return kpis_dict, roc_auc_results, models




# Run the pipeline and save results
if __name__ == "__main__":
    kpis_dict, roc_auc_results, models = model_pipeline()
    joblib.dump({"kpis": kpis_dict, "roc_auc": roc_auc_results, "models": models}, "results.pkl")