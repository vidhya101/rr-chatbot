import React, { useState, useEffect } from 'react';
import { API_BASE_URL, fetchWithErrorHandling, debugLog } from '../utils/config';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Snackbar
} from '@mui/material';
import {
  ModelTraining as ModelIcon,
  Dataset as DatasetIcon,
  Settings as SettingsIcon,
  PlayArrow as StartIcon,
  Assessment as ResultsIcon,
  ExpandMore as ExpandMoreIcon,
  Timeline as TimelineIcon,
  Textsms as TextIcon,
  BubbleChart as ClusterIcon,
  Functions as RegressionIcon,
  Category as ClassificationIcon,
  Refresh as RefreshIcon,
  Upload as UploadIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { Link, useNavigate } from 'react-router-dom';

const DataModeling = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelParameters, setModelParameters] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedVisualizationData, setSelectedVisualizationData] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  const steps = [
    'Select Data',
    'Data Exploration',
    'Data Cleaning',
    'Data Analysis',
    'Visualizations',
    'Dashboard'
  ];

  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [statusPollingInterval, setStatusPollingInterval] = useState(null);

  useEffect(() => {
    fetchFiles();
    return () => {
      if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
      }
    };
  }, []);

  const fetchFiles = async () => {
    setLoading(true);
    try {
      debugLog('Fetching available files for modeling', null);
      
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/data/files`);
      
      debugLog('Files API response received', data);
      
      if (data.success) {
        debugLog(`Loaded ${data.files ? data.files.length : 0} files for modeling`, data.files);
        setUploadedFiles(data.files || []);
        
        // Show message if no files available
        if (!data.files || data.files.length === 0) {
          setSnackbar({
            open: true,
            message: 'No files available. Please upload data files first.',
            severity: 'info'
          });
        }
      }
    } catch (err) {
      console.error('Error fetching files:', err);
      setError('Failed to fetch available files');
      setSnackbar({
        open: true,
        message: 'Failed to fetch available files: ' + (err.message || ''),
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setActiveStep(1);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setActiveStep(2);
  };

  const handleParameterChange = (param, value) => {
    setModelParameters({
      ...modelParameters,
      [param]: value
    });
  };

  const handleTrainModel = async () => {
    if (!selectedFile || !selectedModel) {
      setSnackbar({
        open: true,
        message: 'Please select a file and model first',
        severity: 'warning'
      });
      return;
    }

    setLoading(true);
    setError(null);

    const params = {
      fileId: selectedFile.id,
      modelType: selectedModel,
      parameters: modelParameters
    };

    try {
      debugLog('Training model with parameters', params);
      
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/models/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
      });
      
      debugLog('Training response received', data);

      if (data.success) {
        setModelResults(data.model);
        setActiveStep(4);
        setSnackbar({
          open: true,
          message: 'Model trained successfully',
          severity: 'success'
        });
      }
    } catch (err) {
      console.error('Error training model:', err);
      let errorMessage = 'Failed to train model'; 
      
      // Try to extract meaningful error message
      if (err.response && err.response.data) {
        errorMessage = err.response.data.error || err.response.data.message || errorMessage;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      setSnackbar({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      setSnackbar({
        open: true,
        message: 'Please select a file first',
        severity: 'warning'
      });
      return;
    }

    setLoading(true);
    setError(null);

    try {
      debugLog('Starting data analysis', { fileId: selectedFile.id });
      
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/modeling/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_id: selectedFile.id
        })
      });
      
      debugLog('Analysis response received', data);

      if (data.success) {
        setAnalysisResults(data.results);
        setActiveStep(5); // Move to dashboard view
        setSnackbar({
          open: true,
          message: 'Analysis completed successfully',
          severity: 'success'
        });
      }
    } catch (err) {
      console.error('Error analyzing data:', err);
      let errorMessage = 'Failed to analyze data';
      
      if (err.response && err.response.data) {
        errorMessage = err.response.data.error || err.response.data.message || errorMessage;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      setSnackbar({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Add this new function to handle NaN values
  const handleNaNValues = (data) => {
    if (typeof data === 'object' && data !== null) {
      if (Array.isArray(data)) {
        return data.map(item => handleNaNValues(item));
      } else {
        const result = {};
        for (const key in data) {
          result[key] = handleNaNValues(data[key]);
        }
        return result;
      }
    } else if (typeof data === 'number' && isNaN(data)) {
      return null;
    }
    return data;
  };

  const handlePlotClick = (data, index) => {
    setSelectedVisualizationData(data.points[0]);
    // Update other visualizations based on selection
    if (!analysisResults?.dashboard?.visualizations) return;
    
    const updatedVisualizations = analysisResults.dashboard.visualizations.map((viz, i) => {
      if (i !== index) {
        const plotData = JSON.parse(viz.plot);
        plotData.data = plotData.data.map(trace => ({
          ...trace,
          opacity: selectedVisualizationData ? 0.3 : 1,
          selectedpoints: selectedVisualizationData ? [selectedVisualizationData.pointIndex] : undefined
        }));
        return { ...viz, plot: JSON.stringify(plotData) };
      }
      return viz;
    });
    
    setAnalysisResults(prev => ({
      ...prev,
      dashboard: {
        ...prev.dashboard,
        visualizations: updatedVisualizations
      }
    }));
  };

  const groupVisualizations = (visualizations) => {
    return {
      distribution: visualizations.filter(viz => viz.type === 'distribution'),
      correlation: visualizations.filter(viz => viz.type === 'correlation'),
      trend: visualizations.filter(viz => viz.type === 'trend'),
      scatter: visualizations.filter(viz => viz.type === 'scatter'),
      category: visualizations.filter(viz => viz.type === 'category'),
      '3d': visualizations.filter(viz => viz.type === '3d'),
      parallel: visualizations.filter(viz => viz.type === 'parallel')
    };
  };

  const handleGoToVisualizations = () => {
    if (analysisResults?.dashboard?.visualizations) {
      // Store visualizations in localStorage or state management
      localStorage.setItem('visualizations', JSON.stringify(analysisResults.dashboard.visualizations));
      // Navigate to visualizations component
      navigate('/visualizations');
    } else {
      setSnackbar({
        open: true,
        message: 'Please complete data analysis to view visualizations',
        severity: 'warning'
      });
    }
  };

  const renderFileSelection = () => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Select Data Source
          </Typography>
          <Button 
            variant="outlined" 
            size="small" 
            startIcon={<RefreshIcon />}
            onClick={fetchFiles}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        
        {loading ? (
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        ) : uploadedFiles.length > 0 ? (
          <List>
            {uploadedFiles.map((file) => (
              <ListItem
                button
                key={file.id}
                onClick={() => handleFileSelect(file)}
                selected={selectedFile?.id === file.id}
                sx={{
                  borderRadius: 1,
                  mb: 1,
                  border: '1px solid',
                  borderColor: selectedFile?.id === file.id ? 'primary.main' : 'divider',
                  bgcolor: selectedFile?.id === file.id ? 'primary.light' : 'background.paper',
                  '&:hover': {
                    bgcolor: selectedFile?.id === file.id ? 'primary.light' : 'action.hover',
                  }
                }}
              >
                <ListItemIcon>
                  <DatasetIcon color={selectedFile?.id === file.id ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText
                  primary={file.name || file.original_name}
                  secondary={
                    <>
                      <Typography variant="body2" component="span">
                        Type: {file.type.toUpperCase()}
                      </Typography>
                      <br />
                      <Typography variant="body2" component="span">
                        {file.rows != null ? `${file.rows} rows` : ''} 
                        {file.columns ? `, ${file.columns.length} columns` : ''}
                      </Typography>
                    </>
                  }
                />
              </ListItem>
            ))}
          </List>
        ) : (
          <Box 
            p={3} 
            textAlign="center" 
            border={1} 
            borderColor="divider" 
            borderRadius={1}
            bgcolor="action.hover"
          >
            <Typography variant="body1" color="textSecondary" gutterBottom>
              No data files available
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Please upload data files in the Upload section first
            </Typography>
            <Button 
              variant="contained" 
              component={Link} 
              to="/upload"
              startIcon={<UploadIcon />}
            >
              Go to Upload
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderModelSelection = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Choose Model Type
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <Card
              onClick={() => handleModelSelect('regression')}
              sx={{
                cursor: 'pointer',
                bgcolor: selectedModel === 'regression' ? 'primary.light' : 'background.paper'
              }}
            >
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <RegressionIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Regression</Typography>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Predict continuous values
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Card
              onClick={() => handleModelSelect('classification')}
              sx={{
                cursor: 'pointer',
                bgcolor: selectedModel === 'classification' ? 'primary.light' : 'background.paper'
              }}
            >
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <ClassificationIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Classification</Typography>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Predict categories or classes
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Card
              onClick={() => handleModelSelect('clustering')}
              sx={{
                cursor: 'pointer',
                bgcolor: selectedModel === 'clustering' ? 'primary.light' : 'background.paper'
              }}
            >
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <ClusterIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Clustering</Typography>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Group similar data points
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderParameterConfiguration = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Configure Model Parameters
        </Typography>
        {selectedModel === 'regression' && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <TextField
                  label="Learning Rate"
                  type="number"
                  value={modelParameters.learning_rate || 0.01}
                  onChange={(e) => handleParameterChange('learning_rate', e.target.value)}
                  inputProps={{ step: 0.001, min: 0.001, max: 1 }}
                />
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <TextField
                  label="Number of Epochs"
                  type="number"
                  value={modelParameters.epochs || 100}
                  onChange={(e) => handleParameterChange('epochs', e.target.value)}
                  inputProps={{ step: 1, min: 1 }}
                />
              </FormControl>
            </Grid>
          </Grid>
        )}
        {selectedModel === 'classification' && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Algorithm</InputLabel>
                <Select
                  value={modelParameters.algorithm || 'random_forest'}
                  onChange={(e) => handleParameterChange('algorithm', e.target.value)}
                >
                  <MenuItem value="random_forest">Random Forest</MenuItem>
                  <MenuItem value="svm">Support Vector Machine</MenuItem>
                  <MenuItem value="neural_network">Neural Network</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        )}
        {selectedModel === 'clustering' && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <TextField
                  label="Number of Clusters"
                  type="number"
                  value={modelParameters.n_clusters || 3}
                  onChange={(e) => handleParameterChange('n_clusters', e.target.value)}
                  inputProps={{ step: 1, min: 2 }}
                />
              </FormControl>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );

  const renderModelTraining = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Train Model
        </Typography>
        <Box display="flex" flexDirection="column" alignItems="center" p={3}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleTrainModel}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <StartIcon />}
          >
            {loading ? 'Training...' : 'Start Training'}
          </Button>
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Box>
      </CardContent>
    </Card>
  );

  const renderDataExploration = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Data Exploration
        </Typography>
        {analysisResults?.exploration ? (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Dataset Overview
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Typography>
                  Shape: {handleNaNValues(analysisResults.exploration.shape)[0]} rows × {handleNaNValues(analysisResults.exploration.shape)[1]} columns
                </Typography>
                <Typography>
                  Memory Usage: {handleNaNValues(analysisResults.exploration.memory_usage).toFixed(2)} MB
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Columns
              </Typography>
              <List>
                {handleNaNValues(analysisResults.exploration.columns).map((col) => (
                  <ListItem key={col}>
                    <ListItemText
                      primary={col}
                      secondary={`Type: ${handleNaNValues(analysisResults.exploration.dtypes)[col]}`}
                    />
                    {handleNaNValues(analysisResults.exploration.missing_values)[col] > 0 && (
                      <Chip
                        label={`${handleNaNValues(analysisResults.exploration.missing_values)[col]} missing`}
                        color="warning"
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        ) : (
          <Box display="flex" justifyContent="center" p={3}>
            <Button
              variant="contained"
              onClick={handleStartAnalysis}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <StartIcon />}
            >
              Start Exploration
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderDataCleaning = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Data Cleaning
        </Typography>
        {analysisResults?.cleaning ? (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Cleaning Actions
              </Typography>
              <List>
                {analysisResults.cleaning.actions.map((action, index) => (
                  <ListItem key={index}>
                    {action.action === 'fill_missing' ? (
                      <ListItemText
                        primary={`Filled ${action.count} missing values in ${action.column}`}
                        secondary={`Method: ${action.method}`}
                      />
                    ) : action.action === 'remove_duplicates' ? (
                      <ListItemText
                        primary={`Removed ${action.count} duplicate rows`}
                      />
                    ) : (
                      <ListItemText primary={JSON.stringify(action)} />
                    )}
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        ) : (
          <Box display="flex" justifyContent="center" p={3}>
            <Typography color="textSecondary">
              Complete data exploration to see cleaning results
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderDataAnalysis = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Data Analysis
        </Typography>
        {analysisResults?.analysis ? (
          <Grid container spacing={2}>
            {analysisResults.analysis.insights.map((insight, index) => (
              <Grid item xs={12} key={index}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {insight.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {insight.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        ) : (
          <Box display="flex" justifyContent="center" p={3}>
            <Typography color="textSecondary">
              Complete data cleaning to see analysis results
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderVisualizations = () => {
    if (!analysisResults?.dashboard?.visualizations) {
      return (
        <Box display="flex" justifyContent="center" p={3}>
          <Typography color="textSecondary">
            Complete data analysis to see visualizations
          </Typography>
        </Box>
      );
    }

    // Ensure we have an array of visualizations
    const visualizations = Array.isArray(analysisResults.dashboard.visualizations) 
      ? analysisResults.dashboard.visualizations 
      : [];

    // Debug logging
    console.log('Rendering visualizations:', visualizations.length);
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Interactive Data Visualizations
        </Typography>
        <Grid container spacing={3}>
          {visualizations.map((viz, index) => {
            // Parse the plot data safely
            let plotData, plotLayout;
            try {
              const plotJson = typeof viz.plot === 'string' ? JSON.parse(viz.plot) : viz.plot;
              plotData = plotJson.data || [];
              plotLayout = plotJson.layout || {};
            } catch (err) {
              console.error('Error parsing visualization:', err);
              return null;
            }

            return (
              <Grid item xs={12} md={6} lg={6} key={index}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {viz.title || `Visualization ${index + 1}`}
                    </Typography>
                    <Box height={400} width="100%">
                      <Plot
                        data={plotData}
                        layout={{
                          ...plotLayout,
                          autosize: true,
                          height: 350,
                          width: undefined,
                          margin: { t: 30, r: 10, b: 30, l: 60 },
                          hovermode: 'closest'
                        }}
                        config={{ 
                          responsive: true,
                          displayModeBar: true,
                          scrollZoom: true
                        }}
                        style={{ width: '100%', height: '100%' }}
                        onClick={(data) => handlePlotClick(data, index)}
                        onError={(err) => console.error('Plot error:', err)}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
        {selectedVisualizationData && (
          <Box mt={2}>
            <Alert severity="info" onClose={() => setSelectedVisualizationData(null)}>
              Selected point: {JSON.stringify(selectedVisualizationData)}
            </Alert>
          </Box>
        )}
      </Box>
    );
  };

  const renderDashboard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Advanced Analytics Dashboard
        </Typography>
        {analysisResults?.dashboard ? (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Data Insights & Visualizations
              </Typography>
              <Grid container spacing={2}>
                {Array.isArray(analysisResults.dashboard.visualizations) && 
                 analysisResults.dashboard.visualizations.map((viz, index) => {
                  // Parse the plot data safely
                  let plotData, plotLayout;
                  try {
                    const plotJson = typeof viz.plot === 'string' ? JSON.parse(viz.plot) : viz.plot;
                    plotData = plotJson.data || [];
                    plotLayout = plotJson.layout || {};
                  } catch (err) {
                    console.error('Error parsing dashboard visualization:', err);
                    return null;
                  }

                  return (
                    <Grid item xs={12} md={6} lg={6} key={index}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            {viz.title || `Visualization ${index + 1}`}
                          </Typography>
                          <Box height={400} width="100%">
                            <Plot
                              data={plotData}
                              layout={{
                                ...plotLayout,
                                autosize: true,
                                height: 350,
                                width: undefined,
                                margin: { t: 30, r: 10, b: 30, l: 60 },
                                hovermode: 'closest'
                              }}
                              config={{ 
                                responsive: true,
                                displayModeBar: true,
                                scrollZoom: true
                              }}
                              style={{ width: '100%', height: '100%' }}
                              onError={(err) => console.error('Dashboard plot error:', err)}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
              <Box mt={3} display="flex" justifyContent="center">
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<VisibilityIcon />}
                  onClick={handleGoToVisualizations}
                  size="large"
                >
                  Go to Visualizations
                </Button>
              </Box>
            </Grid>
          </Grid>
        ) : (
          <Box display="flex" justifyContent="center" p={3}>
            <Typography color="textSecondary">
              Complete all analysis steps to view the dashboard
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderCurrentStep = () => {
    switch (activeStep) {
      case 0:
        return renderFileSelection();
      case 1:
        return renderDataExploration();
      case 2:
        return renderDataCleaning();
      case 3:
        return renderDataAnalysis();
      case 4:
        return renderVisualizations();
      case 5:
        return renderDashboard();
      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Data Modeling
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box mb={4}>
        {renderCurrentStep()}
      </Box>

      <Box display="flex" justifyContent="space-between">
        <Button
          onClick={() => setActiveStep((prev) => prev - 1)}
          disabled={activeStep === 0}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={() => setActiveStep((prev) => prev + 1)}
          disabled={activeStep === steps.length - 1 || 
                   (activeStep === 0 && !selectedFile) ||
                   (activeStep === 1 && !selectedModel)}
        >
          Next
        </Button>
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataModeling; 