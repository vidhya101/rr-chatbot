from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
import optuna
from optuna.integration import OptunaSearchCV
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, log_loss,
    explained_variance_score, max_error,
    median_absolute_error, mean_absolute_percentage_error
)

# Additional configuration models
class AdvancedMLConfig(BaseModel):
    enable_model_versioning: bool = True
    track_experiments: bool = True
    save_artifacts: bool = True
    experiment_name: Optional[str] = None

class AutoMLConfig(BaseModel):
    optimization_metric: str
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = -1

class EnsembleConfig(BaseModel):
    ensemble_type: str  # 'voting', 'stacking'
    base_models: List[str]
    weights: Optional[List[float]] = None
    meta_model: Optional[str] = None

class ModelDeploymentConfig(BaseModel):
    deployment_type: str  # 'batch', 'online', 'edge'
    version_name: str
    enable_monitoring: bool = True
    performance_threshold: Optional[float] = None

# Model versioning and experiment tracking setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")

@app.post("/auto_ml", response_model=ModelInfoResponse)
async def perform_auto_ml(dataset_id: str, config: MLTrainingConfig, auto_ml_config: AutoMLConfig):
    """
    Perform automated machine learning using Optuna for hyperparameter optimization
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["data"]
    
    try:
        def objective(trial):
            # Dynamic hyperparameter suggestion based on algorithm
            if config.algorithm == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                }
            elif config.algorithm == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 0.1),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_uniform("subsample", 0.6, 1.0)
                }
            
            # Update config hyperparameters
            config.hyperparameters.update(params)
            
            # Train and evaluate model
            result = await train_model(dataset_id, config)
            
            # Return optimization metric
            return result.performance_metrics[auto_ml_config.optimization_metric]
        
        # Create and run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=auto_ml_config.n_trials,
            timeout=auto_ml_config.timeout,
            n_jobs=auto_ml_config.n_jobs
        )
        
        # Train final model with best parameters
        config.hyperparameters.update(study.best_params)
        final_model = await train_model(dataset_id, config)
        
        return final_model
    
    except Exception as e:
        logger.error(f"Error in AutoML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in AutoML: {str(e)}")

@app.post("/train_ensemble", response_model=ModelInfoResponse)
async def train_ensemble_model(dataset_id: str, config: MLTrainingConfig, ensemble_config: EnsembleConfig):
    """
    Train an ensemble model using multiple base models
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Prepare base models
        base_estimators = []
        for model_name in ensemble_config.base_models:
            if model_name not in ML_ALGORITHMS[config.task_type]:
                raise HTTPException(status_code=400, detail=f"Unsupported base model: {model_name}")
            
            model_class = ML_ALGORITHMS[config.task_type][model_name]
            base_estimators.append((model_name, model_class()))
        
        # Create ensemble model
        if ensemble_config.ensemble_type == "voting":
            if config.task_type == "classification":
                ensemble = VotingClassifier(
                    estimators=base_estimators,
                    weights=ensemble_config.weights,
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=base_estimators,
                    weights=ensemble_config.weights
                )
        
        elif ensemble_config.ensemble_type == "stacking":
            meta_model = None
            if ensemble_config.meta_model:
                meta_model = ML_ALGORITHMS[config.task_type][ensemble_config.meta_model]()
            
            if config.task_type == "classification":
                ensemble = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=5
                )
            else:
                ensemble = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=5
                )
        
        # Update config and train ensemble
        config.algorithm = f"ensemble_{ensemble_config.ensemble_type}"
        ML_ALGORITHMS[config.task_type][config.algorithm] = lambda: ensemble
        
        return await train_model(dataset_id, config)
    
    except Exception as e:
        logger.error(f"Error training ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training ensemble: {str(e)}")

@app.post("/deploy_model")
async def deploy_model(model_id: str, config: ModelDeploymentConfig):
    """
    Deploy a trained model with versioning and monitoring
    """
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_info = models[model_id]
        
        # Log model to MLflow
        with mlflow.start_run(run_name=config.version_name):
            mlflow.log_params(model_info["config"].dict())
            mlflow.log_metrics(model_info["performance_metrics"])
            mlflow.sklearn.log_model(model_info["model"], "model")
            
            # Log feature importance if available
            if model_info.get("feature_importance"):
                mlflow.log_dict(model_info["feature_importance"], "feature_importance.json")
        
        # Set up model monitoring if enabled
        if config.enable_monitoring:
            # Store baseline metrics
            models[model_id]["monitoring"] = {
                "baseline_metrics": model_info["performance_metrics"],
                "performance_threshold": config.performance_threshold,
                "predictions_log": [],
                "drift_metrics": {}
            }
        
        return {
            "message": "Model deployed successfully",
            "model_id": model_id,
            "version": config.version_name,
            "deployment_type": config.deployment_type,
            "monitoring_enabled": config.enable_monitoring
        }
    
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deploying model: {str(e)}")

@app.post("/model_interpretation")
async def interpret_model_predictions(model_id: str, data: List[Dict[str, Any]]):
    """
    Generate model interpretations using SHAP and LIME
    """
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_info = models[model_id]
        model = model_info["model"]
        feature_columns = model_info["config"].feature_columns
        
        # Prepare input data
        X = pd.DataFrame(data)[feature_columns]
        
        # SHAP analysis
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)
        
        # LIME analysis
        categorical_features = [i for i, col in enumerate(feature_columns) 
                             if X[col].dtype == 'object' or X[col].dtype == 'category']
        
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_columns,
            categorical_features=categorical_features,
            mode='regression' if model_info["config"].task_type == 'regression' else 'classification'
        )
        
        # Generate explanations
        interpretations = {
            'global_importance': {
                'shap_values': dict(zip(feature_columns, np.abs(shap_values).mean(0))),
                'permutation_importance': dict(zip(
                    feature_columns,
                    permutation_importance(model, X, model.predict(X), n_repeats=10).importances_mean
                ))
            },
            'local_explanations': []
        }
        
        # Generate local explanations for each instance
        for i in range(len(X)):
            lime_exp = lime_explainer.explain_instance(
                X.iloc[i].values,
                model.predict,
                num_features=len(feature_columns)
            )
            
            interpretations['local_explanations'].append({
                'instance_id': i,
                'feature_values': X.iloc[i].to_dict(),
                'lime_explanation': dict(lime_exp.as_list()),
                'shap_values': dict(zip(feature_columns, shap_values[i]))
            })
        
        return interpretations
    
    except Exception as e:
        logger.error(f"Error interpreting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interpreting model: {str(e)}")

@app.get("/model_performance/{model_id}")
async def get_detailed_performance(model_id: str):
    """
    Get detailed performance metrics and learning curves for a model
    """
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_info = models[model_id]
        model = model_info["model"]
        config = model_info["config"]
        
        # Get original training data
        df = datasets[model_info["dataset_id"]]["data"]
        X = df[config.feature_columns]
        y = df[config.target_column]
        
        # Calculate learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate additional metrics based on task type
        additional_metrics = {}
        if config.task_type == "classification":
            y_pred_proba = model.predict_proba(X)
            precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba[:, 1])
            fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba[:, 1])
            
            additional_metrics.update({
                'average_precision': average_precision_score(y, y_pred_proba[:, 1]),
                'log_loss': log_loss(y, y_pred_proba),
                'pr_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                },
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
            })
        else:
            y_pred = model.predict(X)
            additional_metrics.update({
                'explained_variance': explained_variance_score(y, y_pred),
                'max_error': max_error(y, y_pred),
                'median_absolute_error': median_absolute_error(y, y_pred),
                'mape': mean_absolute_percentage_error(y, y_pred)
            })
        
        return {
            'model_id': model_id,
            'base_metrics': model_info["performance_metrics"],
            'additional_metrics': additional_metrics,
            'learning_curves': {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
                'test_scores_std': np.std(test_scores, axis=1).tolist()
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

# ... rest of the existing code ... 