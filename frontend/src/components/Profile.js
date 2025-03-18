import React, { useState, useRef } from 'react';
import './Profile.css';

// Material UI components
import {
  Paper,
  Typography,
  Avatar,
  Button,
  TextField,
  Grid,
  Divider,
  IconButton,
  Chip,
  Alert,
  Snackbar,
  Card,
  CardContent,
  CardActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge
} from '@mui/material';

// Material UI Icons
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import EmailIcon from '@mui/icons-material/Email';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import ChatIcon from '@mui/icons-material/Chat';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import BarChartIcon from '@mui/icons-material/BarChart';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

// Services
import { updateProfile, uploadProfilePicture } from '../services/userService';

const Profile = ({ darkMode, user }) => {
  const [editing, setEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    bio: user?.bio || '',
    location: user?.location || '',
    website: user?.website || '',
    company: user?.company || ''
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  
  const fileInputRef = useRef(null);

  // Mock data for user stats
  const userStats = {
    conversations: 42,
    files: 15,
    dashboards: 7,
    joinDate: '2023-01-15',
    lastActive: '2023-06-20T14:30:00Z'
  };

  // Mock data for recent activity
  const recentActivity = [
    {
      id: 1,
      type: 'conversation',
      title: 'Conversation about machine learning',
      date: '2023-06-19T10:15:00Z'
    },
    {
      id: 2,
      type: 'file',
      title: 'data_analysis.csv',
      date: '2023-06-18T16:45:00Z'
    },
    {
      id: 3,
      type: 'dashboard',
      title: 'Monthly Analytics',
      date: '2023-06-17T09:30:00Z'
    },
    {
      id: 4,
      type: 'conversation',
      title: 'Python code review',
      date: '2023-06-16T14:20:00Z'
    }
  ];

  // Handle edit mode toggle
  const toggleEditMode = () => {
    if (editing) {
      // Reset form data if canceling edit
      setProfileData({
        name: user?.name || '',
        bio: user?.bio || '',
        location: user?.location || '',
        website: user?.website || '',
        company: user?.company || ''
      });
    }
    setEditing(!editing);
  };

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfileData({
      ...profileData,
      [name]: value
    });
  };

  // Handle profile update
  const handleSaveProfile = async () => {
    try {
      await updateProfile(profileData);
      
      setSnackbar({
        open: true,
        message: 'Profile updated successfully',
        severity: 'success'
      });
      
      setEditing(false);
    } catch (err) {
      console.error('Error updating profile:', err);
      
      setSnackbar({
        open: true,
        message: 'Failed to update profile. Please try again.',
        severity: 'error'
      });
    }
  };

  // Handle profile picture upload
  const handleProfilePictureUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
      const formData = new FormData();
      formData.append('profilePicture', file);
      
      await uploadProfilePicture(formData);
      
      setSnackbar({
        open: true,
        message: 'Profile picture updated successfully',
        severity: 'success'
      });
    } catch (err) {
      console.error('Error uploading profile picture:', err);
      
      setSnackbar({
        open: true,
        message: 'Failed to upload profile picture. Please try again.',
        severity: 'error'
      });
    }
  };

  // Trigger file input click
  const handleAvatarClick = () => {
    fileInputRef.current.click();
  };

  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  // Format time
  const formatTime = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Get activity icon
  const getActivityIcon = (type) => {
    switch (type) {
      case 'conversation':
        return <ChatIcon color="primary" />;
      case 'file':
        return <InsertDriveFileIcon color="secondary" />;
      case 'dashboard':
        return <BarChartIcon style={{ color: '#4caf50' }} />;
      default:
        return <AccessTimeIcon />;
    }
  };

  return (
    <div className={`profile-container ${darkMode ? 'dark-mode' : ''}`}>
      <Paper elevation={0} className="profile-header">
        <Typography variant="h4" className="profile-title">
          Profile
        </Typography>
        
        {!editing ? (
          <Button
            variant="outlined"
            color="primary"
            startIcon={<EditIcon />}
            onClick={toggleEditMode}
            className="edit-profile-button"
          >
            Edit Profile
          </Button>
        ) : (
          <div className="edit-actions">
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<CancelIcon />}
              onClick={toggleEditMode}
              className="cancel-button"
            >
              Cancel
            </Button>
            
            <Button
              variant="contained"
              color="primary"
              startIcon={<SaveIcon />}
              onClick={handleSaveProfile}
              className="save-button"
            >
              Save
            </Button>
          </div>
        )}
      </Paper>
      
      <Grid container spacing={3} className="profile-content">
        <Grid item xs={12} md={4}>
          <Paper className="profile-sidebar">
            <div className="profile-avatar-container">
              <Badge
                overlap="circular"
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                badgeContent={
                  <IconButton 
                    className="avatar-edit-button"
                    onClick={handleAvatarClick}
                  >
                    <PhotoCameraIcon fontSize="small" />
                  </IconButton>
                }
              >
                <Avatar 
                  src={user?.avatar} 
                  alt={user?.name || 'User'} 
                  className="profile-avatar"
                >
                  {user?.name ? user.name.charAt(0).toUpperCase() : 'U'}
                </Avatar>
              </Badge>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                accept="image/*"
                onChange={handleProfilePictureUpload}
              />
            </div>
            
            <Typography variant="h5" className="profile-name">
              {user?.name || 'User'}
            </Typography>
            
            {user?.bio && (
              <Typography variant="body1" className="profile-bio">
                {user.bio}
              </Typography>
            )}
            
            <div className="profile-details">
              {user?.email && (
                <div className="profile-detail-item">
                  <EmailIcon fontSize="small" className="profile-detail-icon" />
                  <Typography variant="body2">{user.email}</Typography>
                </div>
              )}
              
              {user?.location && (
                <div className="profile-detail-item">
                  <CalendarTodayIcon fontSize="small" className="profile-detail-icon" />
                  <Typography variant="body2">Joined {formatDate(userStats.joinDate)}</Typography>
                </div>
              )}
            </div>
            
            <Divider className="profile-divider" />
            
            <div className="profile-stats">
              <div className="stat-item">
                <Typography variant="h6">{userStats.conversations}</Typography>
                <Typography variant="body2">Conversations</Typography>
              </div>
              
              <div className="stat-item">
                <Typography variant="h6">{userStats.files}</Typography>
                <Typography variant="body2">Files</Typography>
              </div>
              
              <div className="stat-item">
                <Typography variant="h6">{userStats.dashboards}</Typography>
                <Typography variant="body2">Dashboards</Typography>
              </div>
            </div>
          </Paper>
          
          <Card className="profile-card recent-activity">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              
              <List>
                {recentActivity.map((activity) => (
                  <ListItem key={activity.id} className="activity-item">
                    <ListItemIcon className="activity-icon">
                      {getActivityIcon(activity.type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={activity.title}
                      secondary={formatTime(activity.date)}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
            
            <CardActions>
              <Button size="small" color="primary">
                View All Activity
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper className="profile-main">
            <Typography variant="h6" gutterBottom>
              {editing ? 'Edit Profile' : 'Profile Information'}
            </Typography>
            
            {editing ? (
              <Grid container spacing={2} className="profile-form">
                <Grid item xs={12}>
                  <TextField
                    label="Name"
                    name="name"
                    value={profileData.name}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    margin="normal"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    label="Bio"
                    name="bio"
                    value={profileData.bio}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    margin="normal"
                    multiline
                    rows={4}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Location"
                    name="location"
                    value={profileData.location}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    margin="normal"
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Company"
                    name="company"
                    value={profileData.company}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    margin="normal"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    label="Website"
                    name="website"
                    value={profileData.website}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    margin="normal"
                  />
                </Grid>
              </Grid>
            ) : (
              <div className="profile-info">
                {user?.bio && (
                  <div className="info-section">
                    <Typography variant="subtitle1" className="info-label">Bio</Typography>
                    <Typography variant="body1">{user.bio}</Typography>
                  </div>
                )}
                
                <div className="info-grid">
                  {user?.location && (
                    <div className="info-item">
                      <Typography variant="subtitle1" className="info-label">Location</Typography>
                      <Typography variant="body1">{user.location}</Typography>
                    </div>
                  )}
                  
                  {user?.company && (
                    <div className="info-item">
                      <Typography variant="subtitle1" className="info-label">Company</Typography>
                      <Typography variant="body1">{user.company}</Typography>
                    </div>
                  )}
                  
                  {user?.website && (
                    <div className="info-item">
                      <Typography variant="subtitle1" className="info-label">Website</Typography>
                      <Typography variant="body1">
                        <a href={user.website} target="_blank" rel="noopener noreferrer">
                          {user.website}
                        </a>
                      </Typography>
                    </div>
                  )}
                  
                  <div className="info-item">
                    <Typography variant="subtitle1" className="info-label">Joined</Typography>
                    <Typography variant="body1">{formatDate(userStats.joinDate)}</Typography>
                  </div>
                  
                  <div className="info-item">
                    <Typography variant="subtitle1" className="info-label">Last Active</Typography>
                    <Typography variant="body1">{formatTime(userStats.lastActive)}</Typography>
                  </div>
                </div>
              </div>
            )}
          </Paper>
          
          <Card className="profile-card usage-stats">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Usage Statistics
              </Typography>
              
              <div className="usage-chips">
                <Chip 
                  icon={<ChatIcon />} 
                  label={`${userStats.conversations} Conversations`} 
                  className="usage-chip"
                />
                <Chip 
                  icon={<InsertDriveFileIcon />} 
                  label={`${userStats.files} Files`} 
                  className="usage-chip"
                />
                <Chip 
                  icon={<BarChartIcon />} 
                  label={`${userStats.dashboards} Dashboards`} 
                  className="usage-chip"
                />
              </div>
              
              <Typography variant="body2" color="textSecondary" className="usage-note">
                View detailed usage statistics in the dashboard section.
              </Typography>
            </CardContent>
            
            <CardActions>
              <Button size="small" color="primary">
                View Dashboard
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default Profile; 