import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './DataAnalysis.css';

// Material UI components
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardHeader,
  CardContent,
  Divider
} from '@mui/material';

// Material UI icons
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PreviewIcon from '@mui/icons-material/Preview';

// TabPanel component for tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`data-analysis-tabpanel-${index}`}
      aria-labelledby={`data-analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const DataAnalysis = ({ darkMode }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [filePath, setFilePath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [modelType, setModelType] = useState('random_forest');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [visualizations, setVisualizations] = useState({});
  
  const navigate = useNavigate();
  
  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
      setSuccess('');
      setPreviewData(null);
      setAnalysisResults(null);
      setDashboardData(null);
    }
  };
  
  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }
    
    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post('/api/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setFilePath(response.data.file_path);
      
      // Show success message
      setSuccess('File uploaded successfully');
      setTimeout(() => setSuccess(''), 3000);
      
      // Generate preview
      await handlePreview(response.data.file_path);
      
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.error || 'Error uploading file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle file preview
  const handlePreview = async (path) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/files/preview', {
        file_path: path
      });
      
      setPreviewData(response.data);
      setColumns(response.data.columns || []);
      
      // If there are columns, set the first numeric column as default target
      if (response.data.columns && response.data.columns.length > 0) {
        // Try to find a numeric column
        const numericColumns = response.data.columns.filter(col => 
          response.data.data_types[col] && 
          (response.data.data_types[col].includes('int') || 
           response.data.data_types[col].includes('float'))
        );
        
        if (numericColumns.length > 0) {
          setTargetColumn(numericColumns[0]);
        } else {
          setTargetColumn(response.data.columns[0]);
        }
      }
      
      // Move to the preview tab
      setActiveTab(1);
      
      // Show success message
      setSuccess('File preview generated successfully');
      setTimeout(() => setSuccess(''), 3000);
      
    } catch (err) {
      console.error('Preview error:', err);
      setError(err.response?.data?.error || 'Error generating preview: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle analysis
  const handleAnalyze = async () => {
    if (!filePath) {
      setError('No file to analyze');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/files/analyze', {
        file_path: filePath,
        target_column: targetColumn,
        model_type: modelType
      });
      
      if (response.data && response.data.results) {
        setAnalysisResults(response.data.results);
        
        // Check if dashboard data is included in the response
        if (response.data.results.dashboard_data) {
          setDashboardData(response.data.results.dashboard_data);
          
          // Parse visualizations
          if (response.data.results.dashboard_data.visualizations) {
            const parsedVisualizations = {};
            Object.entries(response.data.results.dashboard_data.visualizations).forEach(([key, value]) => {
              try {
                parsedVisualizations[key] = typeof value === 'string' ? JSON.parse(value) : value;
              } catch (e) {
                console.error(`Error parsing visualization ${key}:`, e);
              }
            });
            setVisualizations(parsedVisualizations);
          }
        }
        
        // Move to the results tab
        setActiveTab(2);
        
        // Show success message
        setSuccess('Analysis completed successfully');
        setTimeout(() => setSuccess(''), 3000);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.error || 'Error analyzing file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle dashboard creation
  const handleCreateDashboard = async () => {
    if (!filePath) {
      setError('No file to create dashboard for');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/files/dashboard', {
        file_path: filePath,
        target_column: targetColumn
      });
      
      if (response.data && response.data.dashboard_data) {
        setDashboardData(response.data.dashboard_data);
        
        // Parse visualizations
        if (response.data.dashboard_data.visualizations) {
          const parsedVisualizations = {};
          Object.entries(response.data.dashboard_data.visualizations).forEach(([key, value]) => {
            try {
              parsedVisualizations[key] = typeof value === 'string' ? JSON.parse(value) : value;
            } catch (e) {
              console.error(`Error parsing visualization ${key}:`, e);
            }
          });
          setVisualizations(parsedVisualizations);
        }
        
        // Move to the dashboard tab
        setActiveTab(3);
        
        // Show success message
        setSuccess('Dashboard created successfully');
        setTimeout(() => setSuccess(''), 3000);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Dashboard creation error:', err);
      setError(err.response?.data?.error || 'Error creating dashboard: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Render visualizations
  const renderVisualizations = () => {
    if (!visualizations || Object.keys(visualizations).length === 0) {
      return <Alert severity="info" sx={{ mt: 3 }}>No visualizations available</Alert>;
    }
    
    return (
      <div className="visualizations-section">
        <Typography variant="h6" gutterBottom sx={{ mt: 4, mb: 2 }}>Visualizations</Typography>
        
        <Grid container spacing={3}>
          {Object.entries(visualizations).map(([key, plotData], index) => (
            <Grid item xs={12} md={6} key={index}>
              <div className="visualization-container">
                <Typography variant="subtitle1" className="visualization-title">
                  {plotData.layout?.title?.text || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </Typography>
                <Plot
                  data={plotData.data || []}
                  layout={{
                    ...plotData.layout,
                    autosize: true,
                    height: 400,
                    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
                    paper_bgcolor: darkMode ? '#1e1e1e' : '#fff',
                    plot_bgcolor: darkMode ? '#1e1e1e' : '#fff',
                    font: {
                      color: darkMode ? '#f5f5f5' : '#333'
                    }
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            </Grid>
          ))}
        </Grid>
      </div>
    );
  };
  
  // Render data preview
  const renderPreview = () => {
    if (!previewData) {
      return <Alert severity="info">No preview data available</Alert>;
    }
    
    const previewRows = previewData.preview_data || [];
    const columns = previewData.columns || [];
    
    return (
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              {columns.map((column, index) => (
                <TableCell key={index}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {previewRows.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {columns.map((column, colIndex) => (
                  <TableCell key={colIndex}>{row[column]}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };
  
  // Render analysis results
  const renderAnalysisResults = () => {
    if (!analysisResults) {
      return <Alert severity="info">No analysis results available</Alert>;
    }
    
    const { report } = analysisResults;
    
    return (
      <div>
        <Typography variant="h5" gutterBottom>{report.title}</Typography>
        
        <Card sx={{ mb: 3 }}>
          <CardHeader title="Dataset Information" />
          <CardContent>
            <Typography><strong>Name:</strong> {report.dataset_info.name}</Typography>
            <Typography><strong>Rows:</strong> {report.dataset_info.rows}</Typography>
            <Typography><strong>Columns:</strong> {report.dataset_info.columns}</Typography>
            <Typography><strong>Size:</strong> {report.dataset_info.size_mb.toFixed(2)} MB</Typography>
          </CardContent>
        </Card>
        
        <Card sx={{ mb: 3 }}>
          <CardHeader title="Key Insights" />
          <CardContent>
            {report.key_insights.map((insight, index) => (
              <div key={index} className="key-insight">
                <Typography variant="h6">{insight.title}</Typography>
                <Typography paragraph>{insight.description}</Typography>
                {index < report.key_insights.length - 1 && <Divider sx={{ my: 2 }} />}
              </div>
            ))}
          </CardContent>
        </Card>
        
        {dashboardData?.model && (
          <Card sx={{ mb: 3 }}>
            <CardHeader title="Model Performance" />
            <CardContent>
              <Typography><strong>Model Type:</strong> {dashboardData.model.model_type}</Typography>
              <Typography><strong>R² Score:</strong> {dashboardData.model.r2.toFixed(4)}</Typography>
              <Typography><strong>Mean Squared Error:</strong> {dashboardData.model.mse.toFixed(4)}</Typography>
            </CardContent>
          </Card>
        )}
        
        {renderVisualizations()}
      </div>
    );
  };
  
  // Render dashboard
  const renderDashboard = () => {
    if (!dashboardData) {
      return <Alert severity="info">No dashboard data available</Alert>;
    }
    
    return (
      <div>
        <Typography variant="h5" gutterBottom>Interactive Dashboard</Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card className="dashboard-card">
              <CardHeader title="Dataset Summary" />
              <CardContent>
                <Typography><strong>Name:</strong> {dashboardData.dataset_info?.name || fileName}</Typography>
                <Typography><strong>Rows:</strong> {dashboardData.dataset_info?.rows || 'N/A'}</Typography>
                <Typography><strong>Columns:</strong> {dashboardData.dataset_info?.columns || 'N/A'}</Typography>
                <Typography><strong>Size:</strong> {dashboardData.dataset_info?.size_mb?.toFixed(2) || 'N/A'} MB</Typography>
              </CardContent>
            </Card>
          </Grid>
          
          {dashboardData.model && (
            <Grid item xs={12} md={4}>
              <Card className="dashboard-card">
                <CardHeader title="Model Performance" />
                <CardContent>
                  <div className="dashboard-metric">
                    <Typography className="dashboard-metric-label">Model Type:</Typography>
                    <Typography className="dashboard-metric-value">{dashboardData.model.model_type}</Typography>
                  </div>
                  <div className="dashboard-metric">
                    <Typography className="dashboard-metric-label">R² Score:</Typography>
                    <Typography className="dashboard-metric-value">{dashboardData.model.r2?.toFixed(4) || 'N/A'}</Typography>
                  </div>
                  <div className="dashboard-metric">
                    <Typography className="dashboard-metric-label">Mean Squared Error:</Typography>
                    <Typography className="dashboard-metric-value">{dashboardData.model.mse?.toFixed(4) || 'N/A'}</Typography>
                  </div>
                </CardContent>
              </Card>
            </Grid>
          )}
          
          <Grid item xs={12} md={4}>
            <Card className="dashboard-card">
              <CardHeader title="Key Statistics" />
              <CardContent>
                {dashboardData.statistics && Object.entries(dashboardData.statistics).map(([key, value], index) => (
                  <div key={index} className="dashboard-metric">
                    <Typography className="dashboard-metric-label">{key}:</Typography>
                    <Typography className="dashboard-metric-value">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </Typography>
                  </div>
                ))}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        {renderVisualizations()}
      </div>
    );
  };
  
  return (
    <Container maxWidth="lg" className={`data-analysis-container ${darkMode ? 'dark-mode' : ''}`}>
      <Typography variant="h4" gutterBottom>Data Analysis Dashboard</Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={(e, newValue) => setActiveTab(newValue)}
          aria-label="data analysis tabs"
        >
          <Tab 
            label="Upload" 
            icon={<UploadFileIcon />} 
            iconPosition="start" 
            id="data-analysis-tab-0"
            aria-controls="data-analysis-tabpanel-0"
          />
          <Tab 
            label="Preview" 
            icon={<PreviewIcon />} 
            iconPosition="start" 
            id="data-analysis-tab-1"
            aria-controls="data-analysis-tabpanel-1"
            disabled={!previewData}
          />
          <Tab 
            label="Analysis Results" 
            icon={<AnalyticsIcon />} 
            iconPosition="start" 
            id="data-analysis-tab-2"
            aria-controls="data-analysis-tabpanel-2"
            disabled={!analysisResults}
          />
          <Tab 
            label="Dashboard" 
            icon={<DashboardIcon />} 
            iconPosition="start" 
            id="data-analysis-tab-3"
            aria-controls="data-analysis-tabpanel-3"
            disabled={!dashboardData}
          />
        </Tabs>
      </Box>
      
      <TabPanel value={activeTab} index={0}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Upload a Dataset</Typography>
          <Box component="form" noValidate autoComplete="off">
            <Box sx={{ mb: 3 }}>
              <Button
                variant="contained"
                component="label"
                startIcon={<UploadFileIcon />}
              >
                Select File
                <input
                  type="file"
                  hidden
                  onChange={handleFileChange}
                  accept=".csv,.xlsx,.xls,.json,.txt"
                />
              </Button>
              {fileName && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Selected file: {fileName}
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                Supported file types: CSV, Excel, JSON, TXT
              </Typography>
            </Box>
            
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleUpload} 
              disabled={!file || loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}
            >
              {loading ? 'Uploading...' : 'Upload File'}
            </Button>
          </Box>
        </Paper>
      </TabPanel>
      
      <TabPanel value={activeTab} index={1}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Data Preview</Typography>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            renderPreview()
          )}
          
          <Box sx={{ mt: 4 }}>
            <Grid container spacing={3}>
              <Grid item md={6}>
                <FormControl fullWidth>
                  <InputLabel id="target-column-label">Target Column (for predictive modeling)</InputLabel>
                  <Select
                    labelId="target-column-label"
                    id="target-column"
                    value={targetColumn}
                    label="Target Column (for predictive modeling)"
                    onChange={(e) => setTargetColumn(e.target.value)}
                  >
                    <MenuItem value="">
                      <em>None</em>
                    </MenuItem>
                    {columns.map((column, index) => (
                      <MenuItem key={index} value={column}>{column}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item md={6}>
                <FormControl fullWidth>
                  <InputLabel id="model-type-label">Model Type</InputLabel>
                  <Select
                    labelId="model-type-label"
                    id="model-type"
                    value={modelType}
                    label="Model Type"
                    onChange={(e) => setModelType(e.target.value)}
                  >
                    <MenuItem value="linear">Linear Regression</MenuItem>
                    <MenuItem value="random_forest">Random Forest</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleAnalyze} 
                disabled={loading}
                sx={{ mr: 2 }}
                startIcon={loading ? <CircularProgress size={20} /> : <AnalyticsIcon />}
              >
                {loading ? 'Analyzing...' : 'Analyze Data'}
              </Button>
              
              <Button 
                variant="outlined" 
                color="primary" 
                onClick={handleCreateDashboard} 
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <DashboardIcon />}
              >
                {loading ? 'Creating Dashboard...' : 'Create Dashboard'}
              </Button>
            </Box>
          </Box>
        </Paper>
      </TabPanel>
      
      <TabPanel value={activeTab} index={2}>
        <Paper elevation={3} sx={{ p: 3 }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            renderAnalysisResults()
          )}
        </Paper>
      </TabPanel>
      
      <TabPanel value={activeTab} index={3}>
        <Paper elevation={3} sx={{ p: 3 }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            renderDashboard()
          )}
        </Paper>
      </TabPanel>
    </Container>
  );
};

export default DataAnalysis; 