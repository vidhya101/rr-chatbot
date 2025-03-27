import React, { useState, useEffect } from 'react';
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
  Category as ClassificationIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

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
    try {
      const response = await fetch('/api/data/files');
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch files');
      }

      if (data.success) {
        setUploadedFiles(data.files || []);
      }
    } catch (err) {
      console.error('Error fetching files:', err);
      setError('Failed to fetch available files');
      setSnackbar({
        open: true,
        message: 'Failed to fetch available files',
        severity: 'error'
      });
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

    try {
      const response = await fetch('/api/models/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          fileId: selectedFile.id,
          modelType: selectedModel,
          parameters: modelParameters
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to train model');
      }

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
      setError(err.message || 'Failed to train model');
      setSnackbar({
        open: true,
        message: err.message || 'Failed to train model',
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
        <Typography variant="h6" gutterBottom>
          Select Data Source
        </Typography>
        <List>
          {uploadedFiles.map((file) => (
            <ListItem
              button
              key={file.id}
              onClick={() => handleFileSelect(file)}
              selected={selectedFile?.id === file.id}
            >
              <ListItemIcon>
                <DatasetIcon />
              </ListItemIcon>
              <ListItemText
                primary={file.original_name}
                secondary={`Type: ${file.type}, Rows: ${file.rows}, Columns: ${file.columns.length}`}
              />
            </ListItem>
          ))}
        </List>
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
              {modelResults.metrics && Object.entries(modelResults.metrics).map(([metric, value]) => (
                <Box key={metric} display="flex" alignItems="center" mt={1}>
                  <Typography variant="body2" sx={{ minWidth: 120 }}>
                    {metric}:
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