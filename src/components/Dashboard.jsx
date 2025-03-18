import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, FormControl, InputLabel, Select, MenuItem, FormControlLabel, Switch, Toolbar, Tabs, Tab, Snackbar, Alert, Button, Dialog, AppBar, IconButton, Typography } from '@mui/material';
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from 'recharts';
import { styled } from '@mui/material/styles';
import { useApi } from '../contexts/ApiContext';
import { useNotification } from '../contexts/NotificationContext';
import { useDashboard } from '../contexts/DashboardContext';
import DataUpload from './DataUpload';
import DataProcessing from './DataProcessing';
import Visualization from './Visualization';
import MachineLearning from './MachineLearning';
import ChatInterface from './ChatInterface';
import SavedViews from './SavedViews';

// Advanced configuration for real-time updates
const REAL_TIME_UPDATE_INTERVAL = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds

// Enhanced chart configuration
const CHART_CONFIG = {
  animation: {
    duration: 800,
    easing: 'easeInOutQuart'
  },
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    intersect: false,
    mode: 'index'
  },
  plugins: {
    legend: {
      position: 'top',
      labels: {
        usePointStyle: true
      }
    },
    tooltip: {
      enabled: true,
      mode: 'index',
      intersect: false,
      position: 'nearest'
    }
  }
};

// Enhanced styled components
const EnhancedStyledPaper = styled(StyledPaper)(({ theme }) => ({
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '4px',
    background: 'linear-gradient(90deg, #4a90e2, #64b5f6)',
    borderTopLeftRadius: theme.shape.borderRadius,
    borderTopRightRadius: theme.shape.borderRadius
  },
  '&:hover': {
    transform: 'translateY(-4px)',
    transition: 'transform 0.3s ease-in-out',
    boxShadow: theme.shadows[8]
  }
}));

// Enhanced Dashboard Component
const Dashboard = () => {
  const api = useApi();
  const notification = useNotification();
  const dashboard = useDashboard();

  // ... existing state variables ...

  // New state variables for enhanced features
  const [chartAnimations, setChartAnimations] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(REAL_TIME_UPDATE_INTERVAL);
  const [retryCount, setRetryCount] = useState(0);
  const [chartTheme, setChartTheme] = useState('light');
  const [fullScreenMode, setFullScreenMode] = useState(false);
  const [exportFormat, setExportFormat] = useState('png');
  const [dashboardLayout, setDashboardLayout] = useState('grid');
  const [selectedTimeRange, setSelectedTimeRange] = useState('1d');
  const [customDateRange, setCustomDateRange] = useState({ start: null, end: null });
  const [aggregationType, setAggregationType] = useState('sum');
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(false);
  const [annotations, setAnnotations] = useState([]);
  const [savedViews, setSavedViews] = useState([]);
  const [dataFilters, setDataFilters] = useState({});
  const [highlightedSeries, setHighlightedSeries] = useState(null);

  // Enhanced useEffect for real-time updates
  useEffect(() => {
    let intervalId;

    const fetchUpdates = async () => {
      try {
        const response = await api.get('/data/updates', {
          params: {
            timeRange: selectedTimeRange,
            ...customDateRange,
            aggregationType,
            filters: dataFilters
          }
        });

        if (response.data) {
          // Update visualizations with new data
          updateChartData(response.data);
          setRetryCount(0);
        }
      } catch (error) {
        console.error('Error fetching updates:', error);
        
        if (retryCount < MAX_RETRIES) {
          setTimeout(fetchUpdates, RETRY_DELAY);
          setRetryCount(prev => prev + 1);
        } else {
          setError('Failed to fetch updates after multiple attempts');
          setAutoRefresh(false);
        }
      }
    };

    if (autoRefresh) {
      intervalId = setInterval(fetchUpdates, refreshInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [autoRefresh, refreshInterval, selectedTimeRange, customDateRange, aggregationType, dataFilters]);

  // Enhanced chart rendering with animations and interactions
  const renderEnhancedChart = (chartData, type) => {
    const config = {
      ...CHART_CONFIG,
      animation: chartAnimations ? CHART_CONFIG.animation : false,
      theme: chartTheme,
      plugins: {
        ...CHART_CONFIG.plugins,
        annotation: {
          annotations: annotations
        },
        tooltip: {
          ...CHART_CONFIG.plugins.tooltip,
          callbacks: {
            label: (context) => {
              const value = context.raw;
              if (showConfidenceIntervals) {
                const ci = calculateConfidenceInterval(value);
                return `${value} (CI: ${ci.lower.toFixed(2)} - ${ci.upper.toFixed(2)})`;
              }
              return value;
            }
          }
        }
      }
    };

    switch (type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} {...config}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="value"
                fill="#8884d8"
                name={chartData.label || 'Value'}
                onMouseEnter={() => setHighlightedSeries('value')}
                onMouseLeave={() => setHighlightedSeries(null)}
                opacity={highlightedSeries === null || highlightedSeries === 'value' ? 1 : 0.3}
              />
            </BarChart>
          </ResponsiveContainer>
        );
      
      // ... similar enhancements for other chart types ...
    }
  };

  // Enhanced error handling with retry mechanism
  const handleError = async (error, operation) => {
    console.error(`Error during ${operation}:`, error);
    
    if (retryCount < MAX_RETRIES) {
      setRetryCount(prev => prev + 1);
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
      return true; // Retry
    }
    
    setError(`Failed to ${operation} after multiple attempts. Please try again.`);
    return false; // Don't retry
  };

  // Enhanced data processing with confidence intervals
  const calculateConfidenceInterval = (value, confidence = 0.95) => {
    const standardError = Math.sqrt(value) / Math.sqrt(100); // Example calculation
    const zScore = 1.96; // 95% confidence interval
    
    return {
      lower: value - (zScore * standardError),
      upper: value + (zScore * standardError)
    };
  };

  // Enhanced export functionality
  const exportDashboard = async (format = exportFormat) => {
    try {
      const dashboardData = {
        charts: visualizations,
        filters: dataFilters,
        timeRange: selectedTimeRange,
        customDateRange,
        aggregationType,
        annotations
      };

      const response = await api.post('/export', {
        data: dashboardData,
        format
      });

      // Handle the exported file
      const blob = new Blob([response.data], { type: `application/${format}` });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `dashboard_export_${new Date().toISOString()}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      handleError(error, 'export dashboard');
    }
  };

  // Enhanced view management
  const saveCurrentView = async () => {
    try {
      const viewConfig = {
        filters: dataFilters,
        timeRange: selectedTimeRange,
        customDateRange,
        aggregationType,
        annotations,
        layout: dashboardLayout
      };

      const response = await api.post('/views/save', viewConfig);
      setSavedViews([...savedViews, response.data]);
      showNotification('View saved successfully', 'success');
    } catch (error) {
      handleError(error, 'save view');
    }
  };

  const loadSavedView = async (viewId) => {
    try {
      const response = await api.get(`/views/${viewId}`);
      const view = response.data;
      
      setDataFilters(view.filters);
      setSelectedTimeRange(view.timeRange);
      setCustomDateRange(view.customDateRange);
      setAggregationType(view.aggregationType);
      setAnnotations(view.annotations);
      setDashboardLayout(view.layout);
      
      showNotification('View loaded successfully', 'success');
    } catch (error) {
      handleError(error, 'load view');
    }
  };

  // ... rest of the existing code ...

  return (
    <Box sx={{ display: 'flex' }}>
      {/* ... existing AppBar ... */}
      
      {/* Enhanced Main Content */}
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        
        {/* Enhanced Controls */}
        <Paper sx={{ p: 2, mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={selectedTimeRange}
                  onChange={(e) => setSelectedTimeRange(e.target.value)}
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="1d">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                  <MenuItem value="custom">Custom Range</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Aggregation</InputLabel>
                <Select
                  value={aggregationType}
                  onChange={(e) => setAggregationType(e.target.value)}
                >
                  <MenuItem value="sum">Sum</MenuItem>
                  <MenuItem value="avg">Average</MenuItem>
                  <MenuItem value="min">Minimum</MenuItem>
                  <MenuItem value="max">Maximum</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Layout</InputLabel>
                <Select
                  value={dashboardLayout}
                  onChange={(e) => setDashboardLayout(e.target.value)}
                >
                  <MenuItem value="grid">Grid</MenuItem>
                  <MenuItem value="flow">Flow</MenuItem>
                  <MenuItem value="masonry">Masonry</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                }
                label="Auto Refresh"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showConfidenceIntervals}
                    onChange={(e) => setShowConfidenceIntervals(e.target.checked)}
                  />
                }
                label="Show CI"
              />
            </Grid>
          </Grid>
        </Paper>
        
        {/* Enhanced Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
            <Tab label="Upload" />
            <Tab label="Process" />
            <Tab label="Visualize" />
            <Tab label="ML/AI" />
            <Tab label="Chat" />
            <Tab label="Saved Views" />
          </Tabs>
        </Box>
        
        {/* Enhanced Content */}
        <Box sx={{ mt: 2 }}>
          {activeTab === 0 && (
            <EnhancedStyledPaper>
              <DataUpload onDataUploaded={handleDataUploaded} />
            </EnhancedStyledPaper>
          )}
          
          {activeTab === 1 && (
            <EnhancedStyledPaper>
              <DataProcessing dataset={currentDataset} onProcessed={handleDataProcessed} />
            </EnhancedStyledPaper>
          )}
          
          {activeTab === 2 && (
            <EnhancedStyledPaper>
              <Visualization
                dataset={currentDataset}
                chartConfig={CHART_CONFIG}
                showConfidenceIntervals={showConfidenceIntervals}
                annotations={annotations}
                onAnnotationAdd={(annotation) => setAnnotations([...annotations, annotation])}
                onAnnotationRemove={(id) => setAnnotations(annotations.filter(a => a.id !== id))}
                theme={chartTheme}
                onExport={exportDashboard}
              />
            </EnhancedStyledPaper>
          )}
          
          {activeTab === 3 && (
            <EnhancedStyledPaper>
              <MachineLearning dataset={currentDataset} />
            </EnhancedStyledPaper>
          )}
          
          {activeTab === 4 && (
            <EnhancedStyledPaper>
              <ChatInterface dataset={currentDataset} />
            </EnhancedStyledPaper>
          )}
          
          {activeTab === 5 && (
            <EnhancedStyledPaper>
              <SavedViews
                views={savedViews}
                onLoad={loadSavedView}
                onDelete={(viewId) => setSavedViews(savedViews.filter(v => v.id !== viewId))}
                onSave={saveCurrentView}
              />
            </EnhancedStyledPaper>
          )}
        </Box>
      </Box>
      
      {/* Enhanced Notifications */}
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleCloseNotification}
          severity={notification.severity}
          sx={{ width: '100%' }}
          action={
            notification.action && (
              <Button color="inherit" size="small" onClick={notification.action.onClick}>
                {notification.action.label}
              </Button>
            )
          }
        >
          {notification.message}
        </Alert>
      </Snackbar>
      
      {/* Enhanced Full Screen Mode */}
      {fullScreenMode && (
        <Dialog
          fullScreen
          open={true}
          onClose={() => setFullScreenMode(false)}
        >
          <AppBar sx={{ position: 'relative' }}>
            <Toolbar>
              <IconButton
                edge="start"
                color="inherit"
                onClick={() => setFullScreenMode(false)}
                aria-label="close"
              >
                <CloseIcon />
              </IconButton>
              <Typography sx={{ ml: 2, flex: 1 }} variant="h6" component="div">
                Dashboard Full Screen Mode
              </Typography>
              <Button autoFocus color="inherit" onClick={() => setFullScreenMode(false)}>
                Exit
              </Button>
            </Toolbar>
          </AppBar>
          <Box sx={{ p: 3 }}>
            {/* Render current view in full screen */}
            {renderEnhancedChart(currentDataset, chartType)}
          </Box>
        </Dialog>
      )}
    </Box>
  );
};

export default Dashboard; 