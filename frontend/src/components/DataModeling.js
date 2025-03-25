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
  AccordionDetails
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
  const [textInput, setTextInput] = useState('');
  const [textAnalysisResults, setTextAnalysisResults] = useState(null);
  const [timeSeriesConfig, setTimeSeriesConfig] = useState({
    dateColumn: '',
    valueColumn: '',
    forecastPeriods: 30
  });
  const [timeSeriesResults, setTimeSeriesResults] = useState(null);

  const steps = [
    'Select Data',
    'Choose Analysis Type',
    'Configure Parameters',
    'Train/Analyze',
    'View Results'
  ];

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch('/api/data/files');
      const data = await response.json();
      setUploadedFiles(data.files || []);
    } catch (err) {
      console.error('Error fetching files:', err);
      setError('Failed to fetch available files');
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
          parameters: modelParameters,
          targetColumn: modelParameters.targetColumn
        })
      });

      if (!response.ok) throw new Error('Training failed');

      const results = await response.json();
      setModelResults(results);
      setActiveStep(4);
    } catch (err) {
      console.error('Error training model:', err);
      setError('Failed to train model');
    } finally {
      setLoading(false);
    }
  };

  const handleTextAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/analysis/text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: textInput,
          type: 'all'
        })
      });

      if (!response.ok) throw new Error('Text analysis failed');

      const results = await response.json();
      setTextAnalysisResults(results);
      setActiveStep(4);
    } catch (err) {
      console.error('Error analyzing text:', err);
      setError('Failed to analyze text');
    } finally {
      setLoading(false);
    }
  };

  const handleTimeSeriesAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/analysis/time-series', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          fileId: selectedFile.id,
          ...timeSeriesConfig
        })
      });

      if (!response.ok) throw new Error('Time series analysis failed');

      const results = await response.json();
      setTimeSeriesResults(results);
      setActiveStep(4);
    } catch (err) {
      console.error('Error analyzing time series:', err);
      setError('Failed to analyze time series');
    } finally {
      setLoading(false);
    }
  };

  const renderDataSelection = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Select Data Source
        </Typography>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="File Data" />
          <Tab label="Text Input" />
        </Tabs>
        {activeTab === 0 ? (
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
                  primary={file.name}
                  secondary={`Type: ${file.type}, Size: ${(file.size / 1024).toFixed(2)} KB`}
                />
              </ListItem>
            ))}
          </List>
        ) : (
          <Box mt={2}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Enter Text for Analysis"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderAnalysisTypeSelection = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Choose Analysis Type
        </Typography>
        <Grid container spacing={2}>
          {activeTab === 0 ? (
            <>
              <Grid item xs={12} md={4}>
                <Card
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    bgcolor: selectedModel === 'regression' ? 'primary.light' : 'background.paper'
                  }}
                  onClick={() => handleModelSelect('regression')}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <RegressionIcon color="primary" />
                      <Typography variant="h6" ml={1}>
                        Regression
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Predict continuous values
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    bgcolor: selectedModel === 'classification' ? 'primary.light' : 'background.paper'
                  }}
                  onClick={() => handleModelSelect('classification')}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <ClassificationIcon color="primary" />
                      <Typography variant="h6" ml={1}>
                        Classification
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Categorize data into classes
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    bgcolor: selectedModel === 'clustering' ? 'primary.light' : 'background.paper'
                  }}
                  onClick={() => handleModelSelect('clustering')}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <ClusterIcon color="primary" />
                      <Typography variant="h6" ml={1}>
                        Clustering
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Group similar data points
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    bgcolor: selectedModel === 'time_series' ? 'primary.light' : 'background.paper'
                  }}
                  onClick={() => handleModelSelect('time_series')}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <TimelineIcon color="primary" />
                      <Typography variant="h6" ml={1}>
                        Time Series
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Analyze and forecast time series data
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          ) : (
            <Grid item xs={12} md={6}>
              <Card
                variant="outlined"
                sx={{
                  cursor: 'pointer',
                  bgcolor: selectedModel === 'text_analysis' ? 'primary.light' : 'background.paper'
                }}
                onClick={() => handleModelSelect('text_analysis')}
              >
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <TextIcon color="primary" />
                    <Typography variant="h6" ml={1}>
                      Text Analysis
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    Analyze text content using NLP
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );

  const renderParameterConfiguration = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Configure Parameters
        </Typography>
        {selectedModel === 'time_series' ? (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Date Column"
                value={timeSeriesConfig.dateColumn}
                onChange={(e) => setTimeSeriesConfig({
                  ...timeSeriesConfig,
                  dateColumn: e.target.value
                })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Value Column"
                value={timeSeriesConfig.valueColumn}
                onChange={(e) => setTimeSeriesConfig({
                  ...timeSeriesConfig,
                  valueColumn: e.target.value
                })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Forecast Periods"
                value={timeSeriesConfig.forecastPeriods}
                onChange={(e) => setTimeSeriesConfig({
                  ...timeSeriesConfig,
                  forecastPeriods: parseInt(e.target.value)
                })}
              />
            </Grid>
          </Grid>
        ) : (
          <Grid container spacing={2}>
            {getModelParameters(selectedModel).map((param) => (
              <Grid item xs={12} md={6} key={param.name}>
                <TextField
                  fullWidth
                  label={param.label}
                  type={param.type}
                  value={modelParameters[param.name] || param.default}
                  onChange={(e) => handleParameterChange(param.name, e.target.value)}
                  helperText={param.description}
                />
              </Grid>
            ))}
          </Grid>
        )}
      </CardContent>
    </Card>
  );

  const renderAnalysis = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {selectedModel === 'text_analysis' ? 'Text Analysis' : 'Model Training'}
        </Typography>
        <Box display="flex" flexDirection="column" alignItems="center" mt={3}>
          {loading ? (
            <>
              <CircularProgress />
              <Typography variant="body1" mt={2}>
                {selectedModel === 'text_analysis' ? 'Analyzing text...' : 'Training in progress...'}
              </Typography>
            </>
          ) : (
            <Button
              variant="contained"
              color="primary"
              startIcon={<StartIcon />}
              onClick={() => {
                if (selectedModel === 'text_analysis') {
                  handleTextAnalysis();
                } else if (selectedModel === 'time_series') {
                  handleTimeSeriesAnalysis();
                } else {
                  handleTrainModel();
                }
              }}
            >
              {selectedModel === 'text_analysis' ? 'Start Analysis' : 'Start Training'}
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );

  const renderResults = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Analysis Results
        </Typography>
        {selectedModel === 'text_analysis' && textAnalysisResults && (
          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Entities</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  {textAnalysisResults.entities.map((entity, index) => (
                    <ListItem key={index}>
                      <ListItemText primary={entity[0]} secondary={entity[1]} />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Sentiment Analysis</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Label: {textAnalysisResults.sentiment.label}
                  <br />
                  Score: {textAnalysisResults.sentiment.score.toFixed(4)}
                </Typography>
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
        {selectedModel === 'time_series' && timeSeriesResults && (
          <Box>
            <Plot
              data={JSON.parse(timeSeriesResults.plot).data}
              layout={JSON.parse(timeSeriesResults.plot).layout}
            />
          </Box>
        )}
        {modelResults && (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1">
                Model Performance Metrics
              </Typography>
              <List>
                {Object.entries(modelResults.metrics || {}).map(([metric, value]) => (
                  <ListItem key={metric}>
                    <ListItemText
                      primary={metric}
                      secondary={typeof value === 'number' ? value.toFixed(4) : value}
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Advanced Data Analysis
      </Typography>

      <Box mb={3}>
        <Stepper activeStep={activeStep}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box mt={3}>
        {activeStep === 0 && renderDataSelection()}
        {activeStep === 1 && renderAnalysisTypeSelection()}
        {activeStep === 2 && renderParameterConfiguration()}
        {activeStep === 3 && renderAnalysis()}
        {activeStep === 4 && renderResults()}
      </Box>
    </Box>
  );
};

const getModelParameters = (model) => {
  switch (model) {
    case 'regression':
      return [
        {
          name: 'targetColumn',
          label: 'Target Column',
          type: 'text',
          default: '',
          description: 'Column to predict'
        },
        {
          name: 'learning_rate',
          label: 'Learning Rate',
          type: 'number',
          default: 0.01,
          description: 'Step size for gradient descent'
        },
        {
          name: 'epochs',
          label: 'Epochs',
          type: 'number',
          default: 100,
          description: 'Number of training iterations'
        }
      ];
    case 'classification':
      return [
        {
          name: 'targetColumn',
          label: 'Target Column',
          type: 'text',
          default: '',
          description: 'Column to predict'
        },
        {
          name: 'max_depth',
          label: 'Max Depth',
          type: 'number',
          default: 5,
          description: 'Maximum depth of the decision tree'
        },
        {
          name: 'n_estimators',
          label: 'Number of Estimators',
          type: 'number',
          default: 100,
          description: 'Number of trees in the forest'
        }
      ];
    case 'clustering':
      return [
        {
          name: 'n_clusters',
          label: 'Number of Clusters',
          type: 'number',
          default: 3,
          description: 'Number of clusters to form'
        },
        {
          name: 'max_iter',
          label: 'Max Iterations',
          type: 'number',
          default: 300,
          description: 'Maximum number of iterations'
        }
      ];
    default:
      return [];
  }
};

export default DataModeling; 