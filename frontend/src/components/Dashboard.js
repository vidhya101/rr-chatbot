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
  Box
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

  // Fetch dashboards on component mount and when activeTab changes
  useEffect(() => {
    const fetchDashboards = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // In a real app, we would fetch from the API based on the activeTab
        // For now, we'll use mock data
        const response = await getDashboards(activeTab);
        setDashboards(response || mockChartData);
      } catch (err) {
        console.error('Error fetching dashboards:', err);
        setError('Failed to load dashboards. Please try again.');
        setDashboards([]);
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboards();
    
    // Update URL when tab changes
    if (category !== activeTab) {
      navigate(`/dashboard/${activeTab}`);
    }
  }, [activeTab, category, navigate]);

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

  return (
    <div className={`dashboard-container ${darkMode ? 'dark-mode' : ''}`}>
      <Paper elevation={0} className="dashboard-header">
        <Typography variant="h4" className="dashboard-title">
          Dashboards
        </Typography>
        
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => navigate('/dashboard/create')}
          className="create-dashboard-button"
        >
          Create Dashboard
        </Button>
      </Paper>
      
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        indicatorColor="primary"
        textColor="primary"
        className="dashboard-tabs"
      >
        <Tab value="recent" label="Recent" />
        <Tab value="saved" label="Saved" />
        <Tab value="shared" label="Shared with Me" />
        <Tab value="all" label="All Dashboards" />
      </Tabs>
      
      {error && (
        <Alert severity="error" className="dashboard-alert">
          {error}
        </Alert>
      )}
      
      {loading ? (
        <div className="dashboard-loading">
          <CircularProgress />
          <Typography variant="body1">Loading dashboards...</Typography>
        </div>
      ) : dashboards.length === 0 ? (
        <div className="dashboard-empty">
          <Typography variant="h6">No dashboards found</Typography>
          <Typography variant="body1">
            Create a new dashboard to visualize your data.
          </Typography>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => navigate('/dashboard/create')}
            className="create-dashboard-button-empty"
          >
            Create Dashboard
          </Button>
        </div>
      ) : (
        <Grid container spacing={3} className="dashboard-grid">
          {dashboards.map((dashboard) => (
            <Grid item xs={12} sm={6} md={4} key={dashboard.id}>
              <Card className="dashboard-card">
                <CardContent className="dashboard-card-content">
                  <div className="dashboard-card-header">
                    <div className="dashboard-card-icon">
                      {dashboard.icon || getChartIcon(dashboard.type)}
                    </div>
                    <Typography variant="h6" className="dashboard-card-title">
                      {dashboard.title}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, dashboard)}
                      className="dashboard-card-menu"
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </div>
                  
                  <Typography variant="body2" color="textSecondary" className="dashboard-card-description">
                    {dashboard.description}
                  </Typography>
                  
                  <Box className="dashboard-card-chart-placeholder">
                    {/* In a real app, this would be a chart component */}
                    <div className="chart-placeholder">
                      {dashboard.icon || getChartIcon(dashboard.type)}
                    </div>
                  </Box>
                  
                  <div className="dashboard-card-footer">
                    <Chip
                      label={dashboard.type}
                      size="small"
                      className={`dashboard-card-type ${dashboard.type}`}
                    />
                    <Typography variant="caption" color="textSecondary">
                      Updated: {formatDate(dashboard.lastUpdated)}
                    </Typography>
                  </div>
                </CardContent>
                
                <Divider />
                
                <CardActions className="dashboard-card-actions">
                  <Button
                    size="small"
                    startIcon={<EditIcon />}
                    onClick={() => {
                      setSelectedDashboard(dashboard);
                      handleEditDashboard();
                    }}
                  >
                    Edit
                  </Button>
                  <Button
                    size="small"
                    startIcon={<ShareIcon />}
                    onClick={() => {
                      setSelectedDashboard(dashboard);
                      handleShareDashboard();
                    }}
                  >
                    Share
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEditDashboard}>
          <EditIcon fontSize="small" className="menu-icon" />
          Edit
        </MenuItem>
        <MenuItem onClick={handleShareDashboard}>
          <ShareIcon fontSize="small" className="menu-icon" />
          Share
        </MenuItem>
        <MenuItem onClick={handleDeleteDashboard}>
          <DeleteIcon fontSize="small" className="menu-icon" />
          Delete
        </MenuItem>
      </Menu>
    </div>
  );
};

export default Dashboard; 