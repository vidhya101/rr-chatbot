import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Alert,
  Container,
  IconButton,
  Tooltip,
  Dialog,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Fullscreen as FullscreenIcon,
  GetApp as DownloadIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { useNavigate } from 'react-router-dom';

const Visualizations = () => {
  const navigate = useNavigate();
  const [visualizations, setVisualizations] = useState([]);
  const [error, setError] = useState(null);
  const [selectedViz, setSelectedViz] = useState(null);

  useEffect(() => {
    // Load visualizations from localStorage
    try {
      const storedVisualizations = localStorage.getItem('visualizations');
      if (storedVisualizations) {
        setVisualizations(JSON.parse(storedVisualizations));
      } else {
        setError('No visualizations available. Please complete data analysis first.');
      }
    } catch (err) {
      console.error('Error loading visualizations:', err);
      setError('Error loading visualizations. Please try again.');
    }
  }, []);

  const handleBack = () => {
    navigate('/data-modeling');
  };

  const handleFullscreen = (viz) => {
    setSelectedViz(viz);
  };

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        <Box display="flex" alignItems="center" mb={4}>
          <IconButton onClick={handleBack} sx={{ mr: 2 }}>
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4">
            Interactive Data Visualizations
          </Typography>
        </Box>

        {error ? (
          <Alert severity="warning" sx={{ mb: 3 }}>
            {error}
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {visualizations.map((viz, index) => {
              // Parse the plot data safely
              let plotData, plotLayout;
              try {
                plotData = viz.plot.data || [];
                plotLayout = viz.plot.layout || {};
              } catch (err) {
                console.error('Error parsing visualization:', err);
                return null;
              }

              return (
                <Grid item xs={12} md={6} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Typography variant="h6">
                          {viz.title || `Visualization ${index + 1}`}
                        </Typography>
                        <Box>
                          <Tooltip title="View Fullscreen">
                            <IconButton onClick={() => handleFullscreen(viz)}>
                              <FullscreenIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </Box>
                      {viz.insight && (
                        <Typography 
                          variant="body2" 
                          color="text.secondary" 
                          sx={{ mb: 2 }}
                        >
                          {viz.insight}
                        </Typography>
                      )}
                      <Box height={500} width="100%" id={`viz-${viz.title}`}>
                        <Plot
                          data={plotData}
                          layout={{
                            ...plotLayout,
                            autosize: true,
                            height: undefined,
                            width: undefined,
                            margin: { t: 30, r: 10, b: 30, l: 60 },
                            hovermode: 'closest'
                          }}
                          config={{ 
                            responsive: true,
                            displayModeBar: true,
                            scrollZoom: true,
                            toImageButtonOptions: {
                              format: 'png',
                              filename: viz.title,
                              height: 1000,
                              width: 1000,
                              scale: 2
                            },
                            modeBarButtons: [[
                              'toImage',
                              'zoom2d',
                              'pan2d',
                              'select2d',
                              'zoomIn2d',
                              'zoomOut2d',
                              'autoScale2d',
                              'resetScale2d'
                            ]]
                          }}
                          style={{ width: '100%', height: '100%' }}
                          useResizeHandler={true}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        )}

        {/* Fullscreen Modal */}
        {selectedViz && (
          <Dialog
            open={true}
            onClose={() => setSelectedViz(null)}
            maxWidth="xl"
            fullWidth
          >
            <DialogContent>
              <Box height="80vh">
                <Plot
                  data={JSON.parse(selectedViz.plot).data}
                  layout={{
                    ...JSON.parse(selectedViz.plot).layout,
                    autosize: true,
                    height: undefined,
                    width: undefined,
                    margin: { t: 30, r: 10, b: 30, l: 60 }
                  }}
                  config={{ 
                    responsive: true,
                    displayModeBar: true,
                    scrollZoom: true,
                    toImageButtonOptions: {
                      format: 'png',
                      filename: selectedViz.title,
                      height: 1200,
                      width: 1600,
                      scale: 2
                    },
                    modeBarButtons: [[
                      'toImage',
                      'zoom2d',
                      'pan2d',
                      'select2d',
                      'zoomIn2d',
                      'zoomOut2d',
                      'autoScale2d',
                      'resetScale2d'
                    ]]
                  }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              </Box>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedViz(null)}>Close</Button>
            </DialogActions>
          </Dialog>
        )}
      </Box>
    </Container>
  );
};

export default Visualizations; 