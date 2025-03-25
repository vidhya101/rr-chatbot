import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import './Dashboard.css';

// Material UI components
import {
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  IconButton,
  Menu,
  MenuItem,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  TextField,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';

// Material UI Icons
import MoreVertIcon from '@mui/icons-material/MoreVert';
import BarChartIcon from '@mui/icons-material/BarChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import AddIcon from '@mui/icons-material/Add';
import ShareIcon from '@mui/icons-material/Share';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import UploadIcon from '@mui/icons-material/Upload';
import HistoryIcon from '@mui/icons-material/History';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import DatasetIcon from '@mui/icons-material/Dataset';
import VisibilityIcon from '@mui/icons-material/Visibility';

// Services
import { getDashboards, deleteDashboard } from '../services/dashboardService';

// Mock data for charts (in a real app, this would come from an API)
const mockChartData = [
  {
    id: 'sentiment-analysis',
    title: 'Sentiment Analysis',
    type: 'pie',
    icon: <PieChartIcon />,
    description: 'Distribution of sentiment in recent conversations',
    lastUpdated: '2023-06-15T10:30:00Z'
  },
  {
    id: 'conversation-volume',
    title: 'Conversation Volume',
    type: 'bar',
    icon: <BarChartIcon />,
    description: 'Number of conversations over time',
    lastUpdated: '2023-06-14T14:45:00Z'
  },
  {
    id: 'topic-trends',
    title: 'Topic Trends',
    type: 'line',
    icon: <TimelineIcon />,
    description: 'Trending topics in conversations',
    lastUpdated: '2023-06-13T09:15:00Z'
  },
  {
    id: 'word-cloud',
    title: 'Word Cloud',
    type: 'bubble',
    icon: <BubbleChartIcon />,
    description: 'Most common words in conversations',
    lastUpdated: '2023-06-12T16:20:00Z'
  }
];

const Dashboard = ({ darkMode }) => {
  const { category } = useParams();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState(category || 'recent');
  const [dashboards, setDashboards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedDashboard, setSelectedDashboard] = useState(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [modelingDialogOpen, setModelingDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelParameters, setModelParameters] = useState({});
  const [visualizationDialogOpen, setVisualizationDialogOpen] = useState(false);
  const [selectedVisualization, setSelectedVisualization] = useState('');
  const [visualizationParameters, setVisualizationParameters] = useState({});

  // Fetch dashboards on component mount and when activeTab changes
  useEffect(() => {
    const fetchDashboards = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Fetch uploaded files
        const filesResponse = await fetch('/api/data/files');
        const filesData = await filesResponse.json();
        setUploadedFiles(filesData.files || []);
        
        setDashboards(mockChartData);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboards();
  }, [activeTab]);

  // Handle file upload
  const handleFileUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      setUploadedFiles([...uploadedFiles, data]);
      setUploadDialogOpen(false);
      setSelectedFile(null);
    } catch (err) {
      console.error('Error uploading file:', err);
      setError('Failed to upload file. Please try again.');
    }
  };

  // Handle model training
  const handleModelTraining = async () => {
    if (!selectedModel) return;

    try {
      const response = await fetch('/api/data/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: selectedModel,
          parameters: modelParameters
        })
      });

      if (!response.ok) throw new Error('Training failed');

      setModelingDialogOpen(false);
    } catch (err) {
      console.error('Error training model:', err);
      setError('Failed to train model. Please try again.');
    }
  };

  // Handle visualization creation
  const handleVisualizationCreate = async () => {
    if (!selectedVisualization) return;

    try {
      const response = await fetch('/api/data/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          type: selectedVisualization,
          parameters: visualizationParameters
        })
      });

      if (!response.ok) throw new Error('Visualization creation failed');

      setVisualizationDialogOpen(false);
    } catch (err) {
      console.error('Error creating visualization:', err);
      setError('Failed to create visualization. Please try again.');
    }
  };

  // Render upload dialog
  const renderUploadDialog = () => (
    <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
      <DialogTitle>Upload Data File</DialogTitle>
      <DialogContent>
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={(e) => setSelectedFile(e.target.files[0])}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
        <Button onClick={handleFileUpload} disabled={!selectedFile}>
          Upload
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Render modeling dialog
  const renderModelingDialog = () => (
    <Dialog open={modelingDialogOpen} onClose={() => setModelingDialogOpen(false)}>
      <DialogTitle>Train Model</DialogTitle>
      <DialogContent>
        <FormControl fullWidth margin="normal">
          <InputLabel>Model Type</InputLabel>
          <Select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <MenuItem value="regression">Regression</MenuItem>
            <MenuItem value="classification">Classification</MenuItem>
            <MenuItem value="clustering">Clustering</MenuItem>
          </Select>
        </FormControl>
        {/* Add model-specific parameters here */}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setModelingDialogOpen(false)}>Cancel</Button>
        <Button onClick={handleModelTraining} disabled={!selectedModel}>
          Train
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Render visualization dialog
  const renderVisualizationDialog = () => (
    <Dialog open={visualizationDialogOpen} onClose={() => setVisualizationDialogOpen(false)}>
      <DialogTitle>Create Visualization</DialogTitle>
      <DialogContent>
        <FormControl fullWidth margin="normal">
          <InputLabel>Visualization Type</InputLabel>
          <Select
            value={selectedVisualization}
            onChange={(e) => setSelectedVisualization(e.target.value)}
          >
            <MenuItem value="line">Line Chart</MenuItem>
            <MenuItem value="bar">Bar Chart</MenuItem>
            <MenuItem value="scatter">Scatter Plot</MenuItem>
            <MenuItem value="pie">Pie Chart</MenuItem>
          </Select>
        </FormControl>
        {/* Add visualization-specific parameters here */}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setVisualizationDialogOpen(false)}>Cancel</Button>
        <Button onClick={handleVisualizationCreate} disabled={!selectedVisualization}>
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Handle menu open
  const handleMenuOpen = (event, dashboard) => {
    setAnchorEl(event.currentTarget);
    setSelectedDashboard(dashboard);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  // Handle dashboard deletion
  const handleDeleteDashboard = async () => {
    if (!selectedDashboard) return;
    
    try {
      await deleteDashboard(selectedDashboard.id);
      
      // Remove the dashboard from the list
      setDashboards(dashboards.filter(d => d.id !== selectedDashboard.id));
      
      // Close the menu
      handleMenuClose();
    } catch (err) {
      console.error('Error deleting dashboard:', err);
      setError('Failed to delete dashboard. Please try again.');
    }
  };

  // Handle dashboard edit
  const handleEditDashboard = () => {
    if (!selectedDashboard) return;
    
    // Navigate to edit page
    navigate(`/dashboard/edit/${selectedDashboard.id}`);
    
    // Close the menu
    handleMenuClose();
  };

  // Handle dashboard share
  const handleShareDashboard = () => {
    if (!selectedDashboard) return;
    
    // In a real app, this would open a share dialog
    console.log(`Sharing dashboard: ${selectedDashboard.id}`);
    
    // Close the menu
    handleMenuClose();
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  // Get icon for chart type
  const getChartIcon = (type) => {
    switch (type) {
      case 'bar':
        return <BarChartIcon />;
      case 'pie':
        return <PieChartIcon />;
      case 'line':
        return <TimelineIcon />;
      case 'bubble':
        return <BubbleChartIcon />;
      default:
        return <BarChartIcon />;
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
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
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">Data Analytics Dashboard</Typography>
        <Box>
        <Button
            startIcon={<UploadIcon />}
            onClick={() => setUploadDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Upload Data
        </Button>
          <Button
            startIcon={<ModelTrainingIcon />}
            onClick={() => setModelingDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Train Model
          </Button>
          <Button
            startIcon={<BarChartIcon />}
            onClick={() => setVisualizationDialogOpen(true)}
          >
            Create Visualization
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Uploaded Files Section */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Uploaded Files
                    </Typography>
              <List>
                {Object.entries(uploadedFiles).map(([filename, metadata]) => (
                  <ListItem key={filename}>
                    <ListItemIcon>
                      <DatasetIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={filename}
                      secondary={`Uploaded: ${new Date(metadata.upload_time).toLocaleString()}`}
                    />
                    <IconButton onClick={() => {}}>
                      <VisibilityIcon />
                    </IconButton>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Visualizations */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Visualizations
              </Typography>
              <Grid container spacing={2}>
                {dashboards.map((chart) => (
                  <Grid item xs={12} md={6} key={chart.id}>
                    <Card>
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          {chart.icon}
                          <Typography variant="h6" ml={1}>
                            {chart.title}
                  </Typography>
                  </Box>
                        <Typography variant="body2" color="textSecondary">
                          {chart.description}
                    </Typography>
                </CardContent>
                      <CardActions>
                        <Button size="small" startIcon={<EditIcon />}>
                    Edit
                  </Button>
                        <Button size="small" startIcon={<ShareIcon />}>
                    Share
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {renderUploadDialog()}
      {renderModelingDialog()}
      {renderVisualizationDialog()}
    </Box>
  );
};

export default Dashboard; 