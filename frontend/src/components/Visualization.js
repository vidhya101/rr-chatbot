import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  CircularProgress, 
  Button, 
  Card, 
  CardMedia, 
  CardContent, 
  CardActions,
  Snackbar,
  Alert,
  Tabs,
  Tab,
  Table,
  TableContainer,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import DownloadIcon from '@mui/icons-material/Download';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import CloseIcon from '@mui/icons-material/Close';
import { createDashboard } from '../services/apiService';

// Styled components
const VisualizationContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
  backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
}));

const FullScreenModal = styled(Box)(({ theme }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.8)',
  zIndex: 9999,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(2),
}));

const FullScreenImage = styled('img')({
  maxWidth: '90%',
  maxHeight: '90%',
  objectFit: 'contain',
});

const CloseButton = styled(Button)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  color: 'white',
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  '&:hover': {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
}));

const Visualization = ({ fileData, onClose }) => {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fullScreenImage, setFullScreenImage] = useState(null);
  const [stats, setStats] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    if (fileData && fileData.path) {
      generateDashboard(fileData.path);
    }
  }, [fileData]);

  const generateDashboard = async (filePath) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await createDashboard({
        file_path: filePath,
        title: `Dashboard for ${fileData.name}`
      });
      
      if (response.data && response.data.success) {
        setVisualizations(response.data.visualizations || []);
        setStats(response.data.stats || null);
      } else {
        setError(response.data?.error || 'Failed to generate visualizations');
        setSnackbar({
          open: true,
          message: response.data?.error || 'Failed to generate visualizations',
          severity: 'error'
        });
      }
    } catch (err) {
      console.error('Error generating dashboard:', err);
      setError('Error generating visualizations. Please try again.');
      setSnackbar({
        open: true,
        message: 'Error generating visualizations. Please try again.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const generateCustomVisualization = async (type, params = {}) => {
    if (!fileData || !fileData.path) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await createDashboard({
        file_path: fileData.path,
        visualization_type: type,
        params: params
      });
      
      if (response.data && response.data.success) {
        // Add the new visualization to the list
        const newViz = response.data.visualization;
        setVisualizations(prev => [...prev, newViz]);
        
        setSnackbar({
          open: true,
          message: 'Visualization generated successfully',
          severity: 'success'
        });
      } else {
        setError(response.data?.error || 'Failed to generate visualization');
        setSnackbar({
          open: true,
          message: response.data?.error || 'Failed to generate visualization',
          severity: 'error'
        });
      }
    } catch (err) {
      console.error('Error generating visualization:', err);
      setError('Error generating visualization. Please try again.');
      setSnackbar({
        open: true,
        message: 'Error generating visualization. Please try again.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (url, title) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title || 'visualization'}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    setSnackbar({
      open: true,
      message: 'Downloading visualization...',
      severity: 'info'
    });
  };

  const handleFullScreen = (url) => {
    setFullScreenImage(url);
  };

  const closeFullScreen = () => {
    setFullScreenImage(null);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  if (loading && visualizations.length === 0) {
    return (
      <VisualizationContainer>
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
          <CircularProgress />
          <Typography variant="h6" mt={2}>
            Generating visualizations...
          </Typography>
        </Box>
      </VisualizationContainer>
    );
  }

  if (error && visualizations.length === 0) {
    return (
      <VisualizationContainer>
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
          <Typography variant="h6" color="error">
            {error}
          </Typography>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => fileData && generateDashboard(fileData.path)}
            sx={{ mt: 2 }}
          >
            Try Again
          </Button>
        </Box>
      </VisualizationContainer>
    );
  }

  return (
    <VisualizationContainer>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" component="h2">
          Data Visualizations
        </Typography>
        {onClose && (
          <Button 
            variant="outlined" 
            color="primary" 
            startIcon={<CloseIcon />}
            onClick={onClose}
          >
            Close
          </Button>
        )}
      </Box>

      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 2 }}>
        <Tab label="Visualizations" />
        <Tab label="Data Summary" />
      </Tabs>

      {tabValue === 0 && (
        <>
          {loading && (
            <Box display="flex" justifyContent="center" my={2}>
              <CircularProgress size={24} />
            </Box>
          )}

          <Grid container spacing={3}>
            {visualizations.map((viz, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card>
                  <CardMedia
                    component="img"
                    height="200"
                    image={viz.url}
                    alt={viz.title || 'Visualization'}
                    sx={{ objectFit: 'contain', bgcolor: 'white', p: 1 }}
                  />
                  <CardContent>
                    <Typography variant="h6" component="div">
                      {viz.title || 'Visualization'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {viz.description || 'Data visualization'}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button 
                      size="small" 
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownload(viz.url, viz.title)}
                    >
                      Download
                    </Button>
                    <Button 
                      size="small" 
                      startIcon={<FullscreenIcon />}
                      onClick={() => handleFullScreen(viz.url)}
                    >
                      Full Screen
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>

          {visualizations.length === 0 && !loading && (
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
              <Typography variant="body1">
                No visualizations available. Try generating some!
              </Typography>
            </Box>
          )}

          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Generate Custom Visualizations
            </Typography>
            <Grid container spacing={2}>
              <Grid item>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => generateCustomVisualization('histogram')}
                  disabled={loading}
                >
                  Histogram
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => generateCustomVisualization('scatter')}
                  disabled={loading}
                >
                  Scatter Plot
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => generateCustomVisualization('bar')}
                  disabled={loading}
                >
                  Bar Chart
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => generateCustomVisualization('heatmap')}
                  disabled={loading}
                >
                  Correlation Heatmap
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => generateCustomVisualization('boxplot')}
                  disabled={loading}
                >
                  Box Plot
                </Button>
              </Grid>
            </Grid>
          </Box>
        </>
      )}

      {tabValue === 1 && stats && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Data Summary
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                <Typography variant="body2">Rows</Typography>
                <Typography variant="h4">{stats.rows}</Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2, bgcolor: 'secondary.light', color: 'secondary.contrastText' }}>
                <Typography variant="body2">Columns</Typography>
                <Typography variant="h4">{stats.columns}</Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
                <Typography variant="body2">Numeric Columns</Typography>
                <Typography variant="h4">{stats.numeric_columns?.length || 0}</Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2, bgcolor: 'info.light', color: 'info.contrastText' }}>
                <Typography variant="body2">Categorical Columns</Typography>
                <Typography variant="h4">{stats.categorical_columns?.length || 0}</Typography>
              </Paper>
            </Grid>
          </Grid>
          
          <Box mt={4}>
            <Typography variant="subtitle1" gutterBottom>
              Numeric Columns
            </Typography>
            <Grid container spacing={1}>
              {stats.numeric_columns?.map((col, index) => (
                <Grid item key={index}>
                  <Chip label={col} color="primary" variant="outlined" />
                </Grid>
              ))}
              {(!stats.numeric_columns || stats.numeric_columns.length === 0) && (
                <Grid item>
                  <Typography variant="body2" color="text.secondary">
                    No numeric columns found
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>
          
          <Box mt={3}>
            <Typography variant="subtitle1" gutterBottom>
              Categorical Columns
            </Typography>
            <Grid container spacing={1}>
              {stats.categorical_columns?.map((col, index) => (
                <Grid item key={index}>
                  <Chip label={col} color="secondary" variant="outlined" />
                </Grid>
              ))}
              {(!stats.categorical_columns || stats.categorical_columns.length === 0) && (
                <Grid item>
                  <Typography variant="body2" color="text.secondary">
                    No categorical columns found
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>
          
          <Box mt={3}>
            <Typography variant="subtitle1" gutterBottom>
              Column Types
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Column</strong></TableCell>
                    <TableCell><strong>Type</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(stats.column_types || {}).map(([col, type], index) => (
                    <TableRow key={index}>
                      <TableCell>{col}</TableCell>
                      <TableCell>{type}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </Paper>
      )}

      {/* Full screen modal */}
      {fullScreenImage && (
        <FullScreenModal onClick={closeFullScreen}>
          <FullScreenImage 
            src={fullScreenImage} 
            alt="Visualization" 
            onClick={(e) => e.stopPropagation()} 
          />
          <CloseButton 
            variant="contained" 
            startIcon={<CloseIcon />}
            onClick={closeFullScreen}
          >
            Close
          </CloseButton>
        </FullScreenModal>
      )}

      {/* Snackbar for notifications */}
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
    </VisualizationContainer>
  );
};

export default Visualization; 