import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Alert
} from '@mui/material';
import Plot from 'react-plotly.js';

const Visualizations = ({ visualizations, onVisualizationClick }) => {
  if (!visualizations || visualizations.length === 0) {
    return (
      <Box display="flex" justifyContent="center" p={3}>
        <Typography color="textSecondary">
          Complete data analysis to see visualizations
        </Typography>
      </Box>
    );
  }

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
                      onClick={(data) => onVisualizationClick && onVisualizationClick(data, index)}
                      onError={(err) => console.error('Plot error:', err)}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default Visualizations; 