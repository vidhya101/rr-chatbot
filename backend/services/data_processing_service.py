from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import shap
import lime
import lime.lime_tabular
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Additional models for advanced analytics
class AdvancedAnalyticsConfig(BaseModel):
    operation: str
    parameters: Dict[str, Any] = {}

class TimeSeriesConfig(BaseModel):
    time_column: str
    value_column: str
    period: Optional[int] = None

class ClusteringConfig(BaseModel):
    method: str = "kmeans"
    n_clusters: Optional[int] = None
    parameters: Dict[str, Any] = {}

class FeatureSelectionConfig(BaseModel):
    method: str = "mutual_info"
    target_column: str
    n_features: int = 10
    parameters: Dict[str, Any] = {}

class ModelInterpretabilityConfig(BaseModel):
    model_type: str
    target_column: str
    sample_size: int = 100
    parameters: Dict[str, Any] = {}

@app.post("/datasets/{dataset_id}/advanced_analytics", response_model=Dict[str, Any])
async def perform_advanced_analytics(dataset_id: str, config: AdvancedAnalyticsConfig):
    """
    Perform advanced analytics operations on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        results = {}
        
        if config.operation == "advanced_statistics":
            # Perform advanced statistical analysis
            stats_results = {}
            
            # Normality tests
            numeric_cols = df.select_dtypes(include=['number']).columns
            normality_tests = {}
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 3:
                    shapiro_test = stats.shapiro(data)
                    normality_tests[col] = {
                        'shapiro_statistic': float(shapiro_test[0]),
                        'shapiro_p_value': float(shapiro_test[1]),
                        'is_normal': shapiro_test[1] > 0.05
                    }
            stats_results['normality_tests'] = normality_tests
            
            # Correlation analysis with p-values
            if len(numeric_cols) > 1:
                correlation_analysis = {}
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        pearson_corr = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                        spearman_corr = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                        correlation_analysis[f"{col1}_vs_{col2}"] = {
                            'pearson': {
                                'correlation': float(pearson_corr[0]),
                                'p_value': float(pearson_corr[1])
                            },
                            'spearman': {
                                'correlation': float(spearman_corr[0]),
                                'p_value': float(spearman_corr[1])
                            }
                        }
                stats_results['correlation_analysis'] = correlation_analysis
            
            results['advanced_statistics'] = stats_results
            
        elif config.operation == "anomaly_detection":
            # Perform anomaly detection
            method = config.parameters.get('method', 'isolation_forest')
            contamination = config.parameters.get('contamination', 0.1)
            
            numeric_data = df.select_dtypes(include=['number'])
            
            if method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
            else:
                raise HTTPException(status_code=400, detail="Unsupported anomaly detection method")
            
            # Fit detector and predict
            labels = detector.fit_predict(numeric_data)
            scores = detector.score_samples(numeric_data)
            
            # Calculate anomaly statistics
            anomaly_mask = labels == -1
            results['anomaly_detection'] = {
                'total_anomalies': int(sum(anomaly_mask)),
                'anomaly_percentage': float(sum(anomaly_mask) / len(labels) * 100),
                'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                'anomaly_scores': scores.tolist()
            }
        
        return {
            "message": "Advanced analytics completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in advanced analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in advanced analytics: {str(e)}")

@app.post("/datasets/{dataset_id}/time_series_analysis", response_model=Dict[str, Any])
async def analyze_time_series(dataset_id: str, config: TimeSeriesConfig):
    """
    Perform time series analysis on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Ensure time column is datetime
        df[config.time_column] = pd.to_datetime(df[config.time_column])
        ts_df = df.sort_values(config.time_column)
        
        # Perform seasonal decomposition
        ts_data = ts_df.set_index(config.time_column)[config.value_column]
        period = config.period or min(len(ts_data), 12)
        decomposition = seasonal_decompose(ts_data, period=period)
        
        # Perform stationarity test
        adf_test = adfuller(ts_data.dropna())
        
        results = {
            'metrics': {
                'mean': float(ts_data.mean()),
                'std': float(ts_data.std()),
                'trend_direction': 'increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'decreasing',
                'seasonality_strength': float(decomposition.seasonal.std() / ts_data.std()),
                'stationarity': {
                    'adf_statistic': float(adf_test[0]),
                    'p_value': float(adf_test[1]),
                    'is_stationary': adf_test[1] < 0.05
                }
            },
            'components': {
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist()
            }
        }
        
        return {
            "message": "Time series analysis completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in time series analysis: {str(e)}")

@app.post("/datasets/{dataset_id}/clustering", response_model=Dict[str, Any])
async def perform_clustering(dataset_id: str, config: ClusteringConfig):
    """
    Perform clustering analysis on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare numeric data
        numeric_data = df.select_dtypes(include=['number'])
        
        # Scale the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Determine optimal number of clusters if not specified
        if config.method == 'kmeans' and config.n_clusters is None:
            max_clusters = min(10, len(numeric_data) // 2)
            silhouette_scores = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                silhouette_scores.append(silhouette_score(scaled_data, labels))
            n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            n_clusters = config.n_clusters or 2
        
        # Perform clustering
        if config.method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif config.method == 'dbscan':
            clusterer = DBSCAN(eps=config.parameters.get('eps', 0.5),
                             min_samples=config.parameters.get('min_samples', 5))
        else:
            raise HTTPException(status_code=400, detail="Unsupported clustering method")
        
        # Fit clustering model
        labels = clusterer.fit_predict(scaled_data)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        results = {
            'method': config.method,
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'pca_coordinates': pca_result.tolist(),
            'feature_importance': {
                'pc1': dict(zip(numeric_data.columns, pca.components_[0])),
                'pc2': dict(zip(numeric_data.columns, pca.components_[1]))
            }
        }
        
        return {
            "message": "Clustering analysis completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in clustering analysis: {str(e)}")

@app.post("/datasets/{dataset_id}/feature_selection", response_model=Dict[str, Any])
async def select_features(dataset_id: str, config: FeatureSelectionConfig):
    """
    Perform feature selection on the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare data
        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols)
        
        if config.method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=config.n_features)
            selector.fit(X_encoded, y)
            selected_features = pd.DataFrame({
                'feature': X_encoded.columns,
                'importance': selector.scores_
            }).sort_values('importance', ascending=False)
            
        elif config.method == 'rfe':
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator=base_model, n_features_to_select=config.n_features)
            selector.fit(X_encoded, y)
            selected_features = pd.DataFrame({
                'feature': X_encoded.columns,
                'selected': selector.support_,
                'rank': selector.ranking_
            }).sort_values('rank')
        
        results = {
            'method': config.method,
            'selected_features': selected_features.to_dict('records'),
            'n_features_selected': config.n_features
        }
        
        return {
            "message": "Feature selection completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in feature selection: {str(e)}")

@app.post("/datasets/{dataset_id}/model_interpretability", response_model=Dict[str, Any])
async def interpret_model(dataset_id: str, config: ModelInterpretabilityConfig):
    """
    Generate model interpretability insights using SHAP and LIME
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["current"]
    
    try:
        # Prepare data
        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]
        
        # Sample data for interpretation
        if len(X) > config.sample_size:
            X_sample = X.sample(n=config.sample_size, random_state=42)
        else:
            X_sample = X
        
        # Train a simple model for interpretation
        if config.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        model.fit(X, y)
        
        # SHAP Analysis
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        # LIME Analysis
        categorical_features = [i for i, col in enumerate(X.columns) 
                             if col in df.select_dtypes(include=['object', 'category']).columns]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=X.columns,
            class_names=[config.target_column],
            categorical_features=categorical_features,
            mode='regression'
        )
        
        results = {
            'global_importance': dict(zip(X.columns, np.abs(shap_values).mean(0))),
            'sample_explanations': []
        }
        
        # Generate explanations for a few samples
        for i in range(min(5, len(X_sample))):
            lime_exp = lime_explainer.explain_instance(
                X_sample.iloc[i].values,
                model.predict,
                num_features=10
            )
            
            results['sample_explanations'].append({
                'instance_id': i,
                'feature_values': X_sample.iloc[i].to_dict(),
                'prediction': float(model.predict(X_sample.iloc[i:i+1])[0]),
                'lime_explanation': dict(lime_exp.as_list()),
                'shap_values': dict(zip(X.columns, shap_values[i]))
            })
        
        return {
            "message": "Model interpretation completed successfully",
            "dataset_id": dataset_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in model interpretation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in model interpretation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("data_processing_service:app", host="0.0.0.0", port=8001, reload=True) 