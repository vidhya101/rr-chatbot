import React, { useState, useEffect } from 'react';
import './AdminDashboard.css';

// Material UI components
import {
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Tabs,
  Tab,
  Box,
  Divider,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Snackbar,
  Alert
} from '@mui/material';

// Material UI Icons
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import SupervisorAccountIcon from '@mui/icons-material/SupervisorAccount';
import SettingsIcon from '@mui/icons-material/Settings';
import ChatIcon from '@mui/icons-material/Chat';
import DashboardIcon from '@mui/icons-material/Dashboard';
import StorageIcon from '@mui/icons-material/Storage';
import BarChartIcon from '@mui/icons-material/BarChart';
import PeopleIcon from '@mui/icons-material/People';

// Mock data (replace with actual API calls)
const mockUsers = [
  { id: 1, username: 'admin', email: 'admin@example.com', role: 'admin', lastLogin: '2023-03-15T10:30:00Z' },
  { id: 2, username: 'user1', email: 'user1@example.com', role: 'user', lastLogin: '2023-03-14T14:20:00Z' },
  { id: 3, username: 'user2', email: 'user2@example.com', role: 'user', lastLogin: '2023-03-13T09:15:00Z' }
];

const mockStats = {
  totalUsers: 3,
  activeUsers: 2,
  totalChats: 15,
  totalMessages: 120,
  totalFiles: 8,
  totalDashboards: 4
};

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`admin-tabpanel-${index}`}
      aria-labelledby={`admin-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const AdminDashboard = ({ darkMode }) => {
  const [tabValue, setTabValue] = useState(0);
  const [users, setUsers] = useState(mockUsers);
  const [stats, setStats] = useState(mockStats);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // Fetch data on component mount
  useEffect(() => {
    // In a real app, you would fetch data from your API
    // Example:
    // const fetchUsers = async () => {
    //   const response = await api.get('/admin/users');
    //   setUsers(response.data);
    // };
    // fetchUsers();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleEditUser = (user) => {
    setCurrentUser(user);
    setEditDialogOpen(true);
  };

  const handleDeleteUser = (user) => {
    setCurrentUser(user);
    setDeleteDialogOpen(true);
  };

  const handleEditDialogClose = () => {
    setEditDialogOpen(false);
  };

  const handleDeleteDialogClose = () => {
    setDeleteDialogOpen(false);
  };

  const handleUserChange = (e) => {
    const { name, value } = e.target;
    setCurrentUser({ ...currentUser, [name]: value });
  };

  const handleSaveUser = () => {
    // In a real app, you would call your API to update the user
    // Example:
    // await api.put(`/admin/users/${currentUser.id}`, currentUser);
    
    // Update local state
    const updatedUsers = users.map(user => 
      user.id === currentUser.id ? currentUser : user
    );
    setUsers(updatedUsers);
    
    setEditDialogOpen(false);
    setSnackbar({
      open: true,
      message: `User ${currentUser.username} updated successfully`,
      severity: 'success'
    });
  };

  const handleConfirmDelete = () => {
    // In a real app, you would call your API to delete the user
    // Example:
    // await api.delete(`/admin/users/${currentUser.id}`);
    
    // Update local state
    const updatedUsers = users.filter(user => user.id !== currentUser.id);
    setUsers(updatedUsers);
    
    setDeleteDialogOpen(false);
    setSnackbar({
      open: true,
      message: `User ${currentUser.username} deleted successfully`,
      severity: 'success'
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className={`admin-dashboard ${darkMode ? 'dark-mode' : ''}`}>
      <Container maxWidth="lg">
        <Typography variant="h4" className="admin-title">
          Admin Dashboard
        </Typography>
        
        <Paper elevation={3} className="admin-paper">
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
            className="admin-tabs"
          >
            <Tab icon={<DashboardIcon />} label="Overview" />
            <Tab icon={<PeopleIcon />} label="Users" />
            <Tab icon={<ChatIcon />} label="Chats" />
            <Tab icon={<StorageIcon />} label="Files" />
            <Tab icon={<SettingsIcon />} label="Settings" />
          </Tabs>
          
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom>
              System Overview
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.totalUsers}
                    </Typography>
                    <Typography color="textSecondary">
                      Total Users
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.totalChats}
                    </Typography>
                    <Typography color="textSecondary">
                      Total Chats
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.totalFiles}
                    </Typography>
                    <Typography color="textSecondary">
                      Total Files
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.activeUsers}
                    </Typography>
                    <Typography color="textSecondary">
                      Active Users
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.totalMessages}
                    </Typography>
                    <Typography color="textSecondary">
                      Total Messages
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h5" component="div">
                      {stats.totalDashboards}
                    </Typography>
                    <Typography color="textSecondary">
                      Total Dashboards
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Typography variant="h6" gutterBottom className="section-title">
              Quick Actions
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<PersonAddIcon />}
                  fullWidth
                >
                  Add User
                </Button>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<SupervisorAccountIcon />}
                  fullWidth
                >
                  Manage Roles
                </Button>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<BarChartIcon />}
                  fullWidth
                >
                  View Reports
                </Button>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<SettingsIcon />}
                  fullWidth
                >
                  System Settings
                </Button>
              </Grid>
            </Grid>
          </TabPanel>
          
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              User Management
            </Typography>
            
            <Paper elevation={2} className="users-list-container">
              <List>
                {users.map((user) => (
                  <React.Fragment key={user.id}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Typography variant="subtitle1">
                            {user.username} 
                            <span className={`role-badge ${user.role}`}>
                              {user.role}
                            </span>
                          </Typography>
                        }
                        secondary={
                          <>
                            <Typography component="span" variant="body2" color="textPrimary">
                              {user.email}
                            </Typography>
                            <br />
                            <Typography component="span" variant="body2" color="textSecondary">
                              Last login: {formatDate(user.lastLogin)}
                            </Typography>
                          </>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton 
                          edge="end" 
                          aria-label="edit"
                          onClick={() => handleEditUser(user)}
                        >
                          <EditIcon />
                        </IconButton>
                        <IconButton 
                          edge="end" 
                          aria-label="delete"
                          onClick={() => handleDeleteUser(user)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                ))}
              </List>
            </Paper>
          </TabPanel>
          
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" gutterBottom>
              Chat Management
            </Typography>
            <Typography variant="body1">
              This section allows you to manage all chat conversations in the system.
            </Typography>
          </TabPanel>
          
          <TabPanel value={tabValue} index={3}>
            <Typography variant="h6" gutterBottom>
              File Management
            </Typography>
            <Typography variant="body1">
              This section allows you to manage all uploaded files in the system.
            </Typography>
          </TabPanel>
          
          <TabPanel value={tabValue} index={4}>
            <Typography variant="h6" gutterBottom>
              System Settings
            </Typography>
            <Typography variant="body1">
              This section allows you to configure system-wide settings.
            </Typography>
          </TabPanel>
        </Paper>
      </Container>
      
      {/* Edit User Dialog */}
      <Dialog open={editDialogOpen} onClose={handleEditDialogClose}>
        <DialogTitle>Edit User</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            name="username"
            label="Username"
            type="text"
            fullWidth
            value={currentUser?.username || ''}
            onChange={handleUserChange}
          />
          <TextField
            margin="dense"
            name="email"
            label="Email"
            type="email"
            fullWidth
            value={currentUser?.email || ''}
            onChange={handleUserChange}
          />
          <TextField
            margin="dense"
            name="role"
            label="Role"
            type="text"
            fullWidth
            value={currentUser?.role || ''}
            onChange={handleUserChange}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleEditDialogClose} color="primary">
            Cancel
          </Button>
          <Button onClick={handleSaveUser} color="primary">
            Save
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete User Dialog */}
      <Dialog open={deleteDialogOpen} onClose={handleDeleteDialogClose}>
        <DialogTitle>Delete User</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete user "{currentUser?.username}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteDialogClose} color="primary">
            Cancel
          </Button>
          <Button onClick={handleConfirmDelete} color="secondary">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default AdminDashboard; 