import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  IconButton,
  Divider,
  Paper
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  Timeline as LineChartIcon,
  PieChart as PieChartIcon,
  ScatterPlot as ScatterPlotIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Fullscreen as FullscreenIcon
} from '@mui/icons-material';

const Visualization = ({ fileData, onClose }) => {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedVisualization, setSelectedVisualization] = useState(null);
  const [showFullScreen, setShowFullScreen] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    if (fileData?.id) {
      generateVisualizations(fileData.id);
    }
  }, [fileData]);

  const generateVisualizations = async (fileId) => {
    setLoading(true);
    setError(null);
    
    try {
      // Generate basic visualizations
      const visualizationTypes = ['auto', 'scatter', 'line', 'bar', 'histogram'];
      const newVisualizations = [];

      for (const type of visualizationTypes) {
        const response = await fetch('/api/data/visualize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            fileId,
            type,
            columns: {
              x: fileData.columns[0], // Use first column as x by default
              y: fileData.columns[1]  // Use second column as y by default
            }
          })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `Failed to create ${type} visualization`);
        }

        if (data.success) {
          newVisualizations.push({
            ...data.visualization,
            title: `${type.charAt(0).toUpperCase() + type.slice(1)} Chart`,
            description: `${type} visualization of ${fileData.original_name}`
          });
        }
      }

      setVisualizations(newVisualizations);
    } catch (err) {
      console.error('Error generating visualizations:', err);
      setError(err.message || 'Failed to generate visualizations');
      setSnackbar({
        open: true,
        message: err.message || 'Failed to generate visualizations',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleViewVisualization = (visualization) => {
    window.open(`/api/data/visualization/${visualization.id}`, '_blank');
  };

  const handleDeleteVisualization = async (visualization) => {
    try {
      const response = await fetch(`/api/data/visualization/${visualization.id}`, {
        method: 'DELETE'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to delete visualization');
      }

      if (data.success) {
        setVisualizations(prevVisualizations => 
          prevVisualizations.filter(v => v.id !== visualization.id)
        );
        setSnackbar({
          open: true,
          message: 'Visualization deleted successfully',
          severity: 'success'
        });
      }
    } catch (err) {
      console.error('Error deleting visualization:', err);
      setSnackbar({
        open: true,
        message: err.message || 'Failed to delete visualization',
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Visualizations for {fileData?.original_name}
      </Typography>

      <Grid container spacing={3}>
        {visualizations.map((visualization) => (
          <Grid item xs={12} md={6} key={visualization.id}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    {visualization.title}
                  </Typography>
                  <IconButton
                    onClick={() => handleViewVisualization(visualization)}
                    size="small"
                  >
                    <FullscreenIcon />
                  </IconButton>
                </Box>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  {visualization.description}
                </Typography>
                <Box
                  component="iframe"
                  src={`/api/data/visualization/${visualization.id}`}
                  width="100%"
                  height={300}
                  frameBorder="0"
                  sx={{ border: '1px solid #eee', borderRadius: 1 }}
                />
              </CardContent>
              <Divider />
              <CardActions>
                <Button
                  size="small"
                  startIcon={<DeleteIcon />}
                  onClick={() => handleDeleteVisualization(visualization)}
                  color="error"
                >
                  Delete
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

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

export default Visualization; 