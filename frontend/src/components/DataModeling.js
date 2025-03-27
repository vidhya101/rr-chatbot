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
  Upload as UploadIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { Link } from 'react-router-dom';

const DataModeling = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelParameters, setModelParameters] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [activeTab, setActiveTab] = useState(0);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  const steps = [
    'Select Data',
    'Choose Model',
    'Configure Parameters',
    'Train Model',
    'View Results'
  ];

  useEffect(() => {
    fetchFiles();
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

  const renderResults = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Results
        </Typography>
        {modelResults && (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1">
                Model Performance Metrics
              </Typography>
              {modelResults.metrics && Object.entries(modelResults.metrics)
                .filter(([metric, value]) => value !== null)
                .map(([metric, value]) => (
                  <Box key={metric} display="flex" alignItems="center" mt={1}>
                    <Typography variant="body2" sx={{ minWidth: 120 }}>
                      {metric.replace(/_/g, ' ').toUpperCase()}:
                    </Typography>
                    <Typography variant="body1" color="primary">
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </Typography>
                  </Box>
                ))}
            </Grid>
            {modelResults.plots && modelResults.plots.map((plot, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Plot
                  data={plot.data}
                  layout={plot.layout}
                  config={{ responsive: true }}
                />
              </Grid>
            ))}
          </Grid>
        )}
      </CardContent>
    </Card>
  );

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
        {activeStep === 0 && renderFileSelection()}
        {activeStep === 1 && renderModelSelection()}
        {activeStep === 2 && renderParameterConfiguration()}
        {activeStep === 3 && renderModelTraining()}
        {activeStep === 4 && renderResults()}
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