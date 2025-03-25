import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Chip,
  Tabs,
  Tab
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
  Dashboard as DashboardIcon
} from '@mui/icons-material';

const VisualizationView = () => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [visualizations, setVisualizations] = useState([]);
  const [dashboards, setDashboards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [dashboardDialogOpen, setDashboardDialogOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [newVisualization, setNewVisualization] = useState({
    name: '',
    type: 'bar',
    dataFile: '',
    xAxis: '',
    yAxis: '',
    options: {}
  });
  const [newDashboard, setNewDashboard] = useState({
    name: '',
    description: '',
    visualizations: []
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [filesResponse, visualizationsResponse, dashboardsResponse] = await Promise.all([
        fetch('/api/data/files'),
        fetch('/api/visualizations'),
        fetch('/api/dashboards')
      ]);

      if (!filesResponse.ok) throw new Error('Failed to fetch files');
      if (!visualizationsResponse.ok) throw new Error('Failed to fetch visualizations');
      if (!dashboardsResponse.ok) throw new Error('Failed to fetch dashboards');

      const [filesData, visualizationsData, dashboardsData] = await Promise.all([
        filesResponse.json(),
        visualizationsResponse.json(),
        dashboardsResponse.json()
      ]);

      setFiles(filesData.files || []);
      setVisualizations(visualizationsData.visualizations || []);
      setDashboards(dashboardsData.dashboards || []);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateVisualization = async () => {
    try {
      const response = await fetch('/api/visualizations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newVisualization)
      });

      if (!response.ok) throw new Error('Failed to create visualization');

      const data = await response.json();
      setVisualizations([...visualizations, data.visualization]);
      setCreateDialogOpen(false);
      setNewVisualization({
        name: '',
        type: 'bar',
        dataFile: '',
        xAxis: '',
        yAxis: '',
        options: {}
      });
    } catch (err) {
      console.error('Error creating visualization:', err);
      setError('Failed to create visualization');
    }
  };

  const handleCreateDashboard = async () => {
    try {
      const response = await fetch('/api/dashboards', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newDashboard)
      });

      if (!response.ok) throw new Error('Failed to create dashboard');

      const data = await response.json();
      setDashboards([...dashboards, data.dashboard]);
      setDashboardDialogOpen(false);
      setNewDashboard({
        name: '',
        description: '',
        visualizations: []
      });
    } catch (err) {
      console.error('Error creating dashboard:', err);
      setError('Failed to create dashboard');
    }
  };

  const handleDeleteVisualization = async (id) => {
    try {
      const response = await fetch(`/api/visualizations/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete visualization');

      setVisualizations(visualizations.filter(v => v.id !== id));
    } catch (err) {
      console.error('Error deleting visualization:', err);
      setError('Failed to delete visualization');
    }
  };

  const handleDeleteDashboard = async (id) => {
    try {
      const response = await fetch(`/api/dashboards/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete dashboard');

      setDashboards(dashboards.filter(d => d.id !== id));
    } catch (err) {
      console.error('Error deleting dashboard:', err);
      setError('Failed to delete dashboard');
    }
  };

  const getVisualizationIcon = (type) => {
    switch (type.toLowerCase()) {
      case 'bar':
        return <BarChartIcon />;
      case 'line':
        return <LineChartIcon />;
      case 'pie':
        return <PieChartIcon />;
      case 'scatter':
        return <ScatterPlotIcon />;
      default:
        return <BarChartIcon />;
    }
  };

  const renderCreateVisualizationDialog = () => (
    <Dialog
      open={createDialogOpen}
      onClose={() => setCreateDialogOpen(false)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Create New Visualization</DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Name"
              value={newVisualization.name}
              onChange={(e) => setNewVisualization({
                ...newVisualization,
                name: e.target.value
              })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth margin="normal">
              <InputLabel>Type</InputLabel>
              <Select
                value={newVisualization.type}
                onChange={(e) => setNewVisualization({
                  ...newVisualization,
                  type: e.target.value
                })}
              >
                <MenuItem value="bar">Bar Chart</MenuItem>
                <MenuItem value="line">Line Chart</MenuItem>
                <MenuItem value="pie">Pie Chart</MenuItem>
                <MenuItem value="scatter">Scatter Plot</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth margin="normal">
              <InputLabel>Data File</InputLabel>
              <Select
                value={newVisualization.dataFile}
                onChange={(e) => setNewVisualization({
                  ...newVisualization,
                  dataFile: e.target.value
                })}
              >
                {files.map(file => (
                  <MenuItem key={file.id} value={file.id}>
                    {file.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="X Axis"
              value={newVisualization.xAxis}
              onChange={(e) => setNewVisualization({
                ...newVisualization,
                xAxis: e.target.value
              })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Y Axis"
              value={newVisualization.yAxis}
              onChange={(e) => setNewVisualization({
                ...newVisualization,
                yAxis: e.target.value
              })}
              margin="normal"
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
        <Button
          onClick={handleCreateVisualization}
          variant="contained"
          color="primary"
        >
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );

  const renderCreateDashboardDialog = () => (
    <Dialog
      open={dashboardDialogOpen}
      onClose={() => setDashboardDialogOpen(false)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Create New Dashboard</DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Name"
              value={newDashboard.name}
              onChange={(e) => setNewDashboard({
                ...newDashboard,
                name: e.target.value
              })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Description"
              value={newDashboard.description}
              onChange={(e) => setNewDashboard({
                ...newDashboard,
                description: e.target.value
              })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth margin="normal">
              <InputLabel>Visualizations</InputLabel>
              <Select
                multiple
                value={newDashboard.visualizations}
                onChange={(e) => setNewDashboard({
                  ...newDashboard,
                  visualizations: e.target.value
                })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip
                        key={value}
                        label={visualizations.find(v => v.id === value)?.name}
                      />
                    ))}
                  </Box>
                )}
              >
                {visualizations.map(visualization => (
                  <MenuItem key={visualization.id} value={visualization.id}>
                    {visualization.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setDashboardDialogOpen(false)}>Cancel</Button>
        <Button
          onClick={handleCreateDashboard}
          variant="contained"
          color="primary"
        >
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Data Visualization
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="Visualizations" />
          <Tab label="Dashboards" />
        </Tabs>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {activeTab === 0 && (
        <>
          <Box display="flex" justifyContent="flex-end" mb={2}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
            >
              Create Visualization
            </Button>
          </Box>

          <Grid container spacing={2}>
            {visualizations.map(visualization => (
              <Grid item xs={12} md={6} lg={4} key={visualization.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      {getVisualizationIcon(visualization.type)}
                      <Typography variant="h6" sx={{ ml: 1 }}>
                        {visualization.name}
                      </Typography>
                    </Box>
                    <Typography color="textSecondary" gutterBottom>
                      Type: {visualization.type}
                    </Typography>
                    <Typography color="textSecondary" gutterBottom>
                      Data: {files.find(f => f.id === visualization.dataFile)?.name}
                    </Typography>
                    <Box mt={2} display="flex" justifyContent="flex-end">
                      <IconButton onClick={() => handleDeleteVisualization(visualization.id)}>
                        <DeleteIcon />
                      </IconButton>
                      <IconButton>
                        <EditIcon />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}

      {activeTab === 1 && (
        <>
          <Box display="flex" justifyContent="flex-end" mb={2}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={() => setDashboardDialogOpen(true)}
            >
              Create Dashboard
            </Button>
          </Box>

          <Grid container spacing={2}>
            {dashboards.map(dashboard => (
              <Grid item xs={12} md={6} key={dashboard.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <DashboardIcon />
                      <Typography variant="h6" sx={{ ml: 1 }}>
                        {dashboard.name}
                      </Typography>
                    </Box>
                    <Typography color="textSecondary" gutterBottom>
                      {dashboard.description}
                    </Typography>
                    <Typography variant="subtitle2" gutterBottom>
                      Visualizations:
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                      {dashboard.visualizations.map(vizId => (
                        <Chip
                          key={vizId}
                          label={visualizations.find(v => v.id === vizId)?.name}
                          size="small"
                        />
                      ))}
                    </Box>
                    <Box display="flex" justifyContent="flex-end">
                      <IconButton onClick={() => handleDeleteDashboard(dashboard.id)}>
                        <DeleteIcon />
                      </IconButton>
                      <IconButton>
                        <EditIcon />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}

      {renderCreateVisualizationDialog()}
      {renderCreateDashboardDialog()}
    </Box>
  );
};

export default VisualizationView; 