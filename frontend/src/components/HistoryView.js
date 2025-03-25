import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tab,
  Tabs
} from '@mui/material';
import {
  History as HistoryIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  GetApp as DownloadIcon,
  Share as ShareIcon,
  ModelTraining as ModelIcon,
  BarChart as ChartIcon,
  Dataset as DatasetIcon
} from '@mui/icons-material';

const HistoryView = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    fetchHistory();
  }, [activeTab]);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const type = getHistoryType(activeTab);
      const response = await fetch(`/api/history/${type}`);
      if (!response.ok) throw new Error('Failed to fetch history');

      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError('Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      const type = getHistoryType(activeTab);
      const response = await fetch(`/api/history/${type}/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete item');

      // Remove item from state
      setHistory(history.filter(item => item.id !== id));
    } catch (err) {
      console.error('Error deleting item:', err);
      setError('Failed to delete item');
    }
  };

  const handleDownload = async (item) => {
    try {
      const response = await fetch(`/api/history/download/${item.id}`);
      if (!response.ok) throw new Error('Failed to download item');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = item.filename || 'download';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading item:', err);
      setError('Failed to download item');
    }
  };

  const handleShare = async (item) => {
    try {
      const response = await fetch('/api/history/share', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id: item.id })
      });

      if (!response.ok) throw new Error('Failed to share item');

      const data = await response.json();
      // Handle share link/success
    } catch (err) {
      console.error('Error sharing item:', err);
      setError('Failed to share item');
    }
  };

  const getHistoryType = (tab) => {
    switch (tab) {
      case 0:
        return 'all';
      case 1:
        return 'models';
      case 2:
        return 'visualizations';
      case 3:
        return 'datasets';
      default:
        return 'all';
    }
  };

  const getItemIcon = (type) => {
    switch (type) {
      case 'model':
        return <ModelIcon />;
      case 'visualization':
        return <ChartIcon />;
      case 'dataset':
        return <DatasetIcon />;
      default:
        return <HistoryIcon />;
    }
  };

  const renderViewDialog = () => (
    <Dialog
      open={viewDialogOpen}
      onClose={() => setViewDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        {selectedItem?.name}
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="subtitle1">Details</Typography>
            <Typography variant="body2">
              Type: {selectedItem?.type}
            </Typography>
            <Typography variant="body2">
              Created: {new Date(selectedItem?.created_at).toLocaleString()}
            </Typography>
            <Typography variant="body2">
              Last Modified: {new Date(selectedItem?.updated_at).toLocaleString()}
            </Typography>
          </Grid>
          {selectedItem?.type === 'model' && (
            <Grid item xs={12}>
              <Typography variant="subtitle1">Model Information</Typography>
              <Typography variant="body2">
                Accuracy: {selectedItem?.metrics?.accuracy}
              </Typography>
              <Typography variant="body2">
                Parameters: {JSON.stringify(selectedItem?.parameters)}
              </Typography>
            </Grid>
          )}
          {selectedItem?.type === 'visualization' && (
            <Grid item xs={12}>
              <Typography variant="subtitle1">Visualization Preview</Typography>
              {/* Add visualization preview here */}
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
        <Button
          startIcon={<DownloadIcon />}
          onClick={() => handleDownload(selectedItem)}
        >
          Download
        </Button>
        <Button
          startIcon={<ShareIcon />}
          onClick={() => handleShare(selectedItem)}
        >
          Share
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
        History
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="All" />
          <Tab label="Models" />
          <Tab label="Visualizations" />
          <Tab label="Datasets" />
        </Tabs>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <List>
            {history.map((item, index) => (
              <React.Fragment key={item.id}>
                <ListItem>
                  <ListItemIcon>
                    {getItemIcon(item.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.name}
                    secondary={
                      <>
                        <Typography variant="body2" color="textSecondary">
                          Created: {new Date(item.created_at).toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Type: {item.type}
                        </Typography>
                      </>
                    }
                  />
                  <Box>
                    <IconButton
                      onClick={() => {
                        setSelectedItem(item);
                        setViewDialogOpen(true);
                      }}
                    >
                      <ViewIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDownload(item)}>
                      <DownloadIcon />
                    </IconButton>
                    <IconButton onClick={() => handleShare(item)}>
                      <ShareIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(item.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </ListItem>
                {index < history.length - 1 && <Divider />}
              </React.Fragment>
            ))}
            {history.length === 0 && (
              <ListItem>
                <ListItemText
                  primary="No items found"
                  secondary="Your history will appear here"
                />
              </ListItem>
            )}
          </List>
        </CardContent>
      </Card>

      {renderViewDialog()}
    </Box>
  );
};

export default HistoryView; 