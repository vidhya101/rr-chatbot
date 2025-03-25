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
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from transformers import pipeline
import spacy
import gensim
from gensim.models import Word2Vec
import networkx as nx
from scipy import stats
import logging
import json
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP models
try:
    nlp = spacy.load('en_core_web_sm')
    sentiment_analyzer = pipeline('sentiment-analysis')
except Exception as e:
    logger.warning(f"Error loading NLP models: {str(e)}")

class MLService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def preprocess_data(self, df, target_col=None, categorical_cols=None, numerical_cols=None):
        """Advanced data preprocessing"""
        try:
            # Automatically identify column types if not specified
            if categorical_cols is None:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if numerical_cols is None:
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                
            # Handle missing values
            for col in numerical_cols:
                df[col].fillna(df[col].median(), inplace=True)
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
            # Encode categorical variables
            for col in categorical_cols:
                if col != target_col:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    self.encoders[col] = encoder
                    
            # Scale numerical features
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['standard'] = scaler
            
            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def train_model(self, df, target_col, model_type='auto', params=None):
        """Train a machine learning model on the given data.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column name
            model_type (str): Type of model to train ('auto', 'classification', 'regression')
            params (dict): Model hyperparameters
            
        Returns:
            Trained model object
        """
        try:
            # Prepare data
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model selection and training
            if model_type == 'auto':
                # Try multiple models and select the best
                models = {
                    'xgboost': xgb.XGBRegressor(),
                    'lightgbm': lgb.LGBMRegressor(),
                    'catboost': CatBoostRegressor(verbose=False)
                }
                
                best_score = float('-inf')
                best_model = None
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                self.models['best'] = best_model
                return {
                    'model_type': 'auto',
                    'best_model': type(best_model).__name__,
                    'score': best_score
                }
            else:
                # Train specific model
                model = self._get_model(model_type, params)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                self.models[model_type] = model
                return {
                    'model_type': model_type,
                    'score': score
                }
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    async def train_model_async(self, dataset_id, config):
        """Asynchronous version of train_model that handles dataset loading.
        
        Args:
            dataset_id (str): ID of the dataset to load
            config (MLTrainingConfig): Training configuration
            
        Returns:
            Trained model object
        """
        # Load dataset
        df = await load_dataset(dataset_id)
        
        # Train model
        return self.train_model(df, config.target_column, config.model_type, config.hyperparameters)

    async def perform_auto_ml(self, dataset_id: str, config: MLTrainingConfig, auto_ml_config: AutoMLConfig):
        """Perform automated machine learning optimization.
        
        Args:
            dataset_id (str): ID of the dataset to use
            config (MLTrainingConfig): Base training configuration
            auto_ml_config (AutoMLConfig): AutoML configuration
            
        Returns:
            Best model and its configuration
        """
        try:
            # Create and run Optuna study
            study = optuna.create_study(direction="maximize")
            
            # Define objective function
            async def objective(trial):
                # Get hyperparameter suggestions
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
                }
                
                # Update config hyperparameters
                config.hyperparameters.update(params)
                
                # Train and evaluate model
                result = await self.train_model_async(dataset_id, config)
                
                # Return optimization metric
                return result.performance_metrics[auto_ml_config.optimization_metric]
            
            # Run optimization
            study.optimize(objective, n_trials=auto_ml_config.n_trials, 
                         timeout=auto_ml_config.timeout,
                         n_jobs=auto_ml_config.n_jobs)
            
            # Get best parameters and train final model
            config.hyperparameters.update(study.best_params)
            final_model = await self.train_model_async(dataset_id, config)
            
            return final_model
            
        except Exception as e:
            logger.error(f"AutoML optimization failed: {str(e)}")
            raise
            
    def analyze_text(self, text, analysis_type='all'):
        """Advanced NLP analysis"""
        try:
            results = {}
            
            if analysis_type in ['all', 'basic']:
                doc = nlp(text)
                results['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
                results['tokens'] = [token.text for token in doc]
                results['pos_tags'] = [(token.text, token.pos_) for token in doc]
                
            if analysis_type in ['all', 'sentiment']:
                sentiment = sentiment_analyzer(text)[0]
                results['sentiment'] = {
                    'label': sentiment['label'],
                    'score': sentiment['score']
                }
                
            if analysis_type in ['all', 'topics']:
                # Topic modeling using gensim
                tokens = [token.text.lower() for token in nlp(text) if not token.is_stop]
                model = Word2Vec([tokens], vector_size=100, window=5, min_count=1)
                results['word_vectors'] = {word: model.wv[word].tolist() for word in model.wv.index_to_key[:10]}
                
            return results
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            raise
            
    def create_advanced_visualization(self, df, viz_type, params=None):
        """Create advanced visualizations"""
        try:
            if viz_type == 'correlation_matrix':
                corr = df.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu'
                ))
                
            elif viz_type == 'distribution':
                fig = go.Figure()
                for col in df.select_dtypes(include=['int64', 'float64']).columns:
                    fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.7))
                    
            elif viz_type == 'time_series':
                if params and 'date_col' in params and 'value_col' in params:
                    fig = px.line(df, x=params['date_col'], y=params['value_col'])
                    
                    # Add trend line
                    x = np.arange(len(df))
                    y = df[params['value_col']]
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(x=df[params['date_col']], y=p(x), name='Trend'))
                    
            elif viz_type == 'network':
                G = nx.from_pandas_edgelist(df, params['source_col'], params['target_col'])
                pos = nx.spring_layout(G)
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                fig = go.Figure(data=[
                    go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'),
                    go.Scatter(x=[pos[node][0] for node in G.nodes()],
                             y=[pos[node][1] for node in G.nodes()],
                             mode='markers+text',
                             text=list(G.nodes()),
                             hoverinfo='text')
                ])
                
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
                
            fig.update_layout(
                title=f"{viz_type.replace('_', ' ').title()} Visualization",
                template="plotly_dark" if params and params.get('dark_mode') else "plotly_white"
            )
            
            return fig.to_json()
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise
            
    def forecast_time_series(self, df, date_col, value_col, periods=30):
        """Time series forecasting using Prophet"""
        try:
            # Prepare data for Prophet
            df_prophet = df[[date_col, value_col]].copy()
            df_prophet.columns = ['ds', 'y']
            
            # Create and fit model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode='multiplicative'
            )
            model.fit(df_prophet)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Create visualization
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=df_prophet['ds'],
                y=df_prophet['y'],
                name='Actual',
                mode='markers+lines'
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                mode='lines'
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Lower Bound'
            ))
            
            fig.update_layout(
                title='Time Series Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            return {
                'forecast': forecast.to_dict('records'),
                'plot': fig.to_json()
            }
        except Exception as e:
            logger.error(f"Error in time series forecasting: {str(e)}")
            raise
            
    def _get_model(self, model_type, params=None):
        """Helper function to get model instance"""
        if params is None:
            params = {}
            
        models = {
            'xgboost': xgb.XGBRegressor(**params),
            'lightgbm': lgb.LGBMRegressor(**params),
            'catboost': CatBoostRegressor(**params),
            'neural_network': tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        }
        
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return models[model_type]

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
        async def objective(trial):
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
            result = await self.train_model_async(dataset_id, config)
            
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
        final_model = await self.train_model_async(dataset_id, config)
        
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