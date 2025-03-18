import React, { useState, useEffect } from 'react';
import './Settings.css';

// Material UI components
import {
  Container,
  Typography,
  Paper,
  Button,
  Tabs,
  Tab,
  Slider,
  Select,
  MenuItem,
  FormControl,
  TextField,
  Divider,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Snackbar,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Switch
} from '@mui/material';

// Material UI Icons
import DarkModeIcon from '@mui/icons-material/DarkMode';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SecurityIcon from '@mui/icons-material/Security';
import LanguageIcon from '@mui/icons-material/Language';
import StorageIcon from '@mui/icons-material/Storage';
import DeleteIcon from '@mui/icons-material/Delete';
import SaveIcon from '@mui/icons-material/Save';
import PersonIcon from '@mui/icons-material/Person';
import VpnKeyIcon from '@mui/icons-material/VpnKey';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import SettingsIcon from '@mui/icons-material/Settings';
import DataUsageIcon from '@mui/icons-material/DataUsage';

// Services
import { updateUserSettings, deleteAccount } from '../services/userService';

const Settings = ({ darkMode, toggleDarkMode, user }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [settings, setSettings] = useState({
    notifications: {
      emailNotifications: true,
      chatCompletions: true,
      marketingEmails: false
    },
    privacy: {
      saveHistory: true,
      shareUsageData: true,
      allowAnalytics: true
    },
    appearance: {
      darkMode: darkMode,
      fontSize: 16,
      messageSpacing: 'normal'
    },
    language: 'en',
    defaultModel: 'digitalogy',
    maxTokens: 2048,
    temperature: 0.7
  });
  
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Handle settings change
  const handleSettingChange = (category, setting, value) => {
    setSettings(prevSettings => ({
      ...prevSettings,
      [category]: {
        ...prevSettings[category],
        [setting]: value
      }
    }));
    
    // Special case for dark mode
    if (category === 'appearance' && setting === 'darkMode') {
      toggleDarkMode();
    }
  };

  // Handle direct setting change (not nested)
  const handleDirectSettingChange = (setting, value) => {
    setSettings(prevSettings => ({
      ...prevSettings,
      [setting]: value
    }));
  };

  // Handle save settings
  const handleSaveSettings = async () => {
    try {
      await updateUserSettings(settings);
      
      setSnackbar({
        open: true,
        message: 'Settings saved successfully',
        severity: 'success'
      });
    } catch (err) {
      console.error('Error saving settings:', err);
      
      setSnackbar({
        open: true,
        message: 'Failed to save settings. Please try again.',
        severity: 'error'
      });
    }
  };

  // Handle delete account
  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== 'DELETE') {
      setSnackbar({
        open: true,
        message: 'Please type DELETE to confirm account deletion',
        severity: 'error'
      });
      return;
    }
    
    try {
      await deleteAccount();
      
      setSnackbar({
        open: true,
        message: 'Account deleted successfully',
        severity: 'success'
      });
      
      // Redirect to login page or home page after account deletion
      // This would typically be handled by a higher-level component
      window.location.href = '/';
    } catch (err) {
      console.error('Error deleting account:', err);
      
      setSnackbar({
        open: true,
        message: 'Failed to delete account. Please try again.',
        severity: 'error'
      });
    } finally {
      setDeleteDialogOpen(false);
      setDeleteConfirmText('');
    }
  };

  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Render tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // General
        return (
          <div className="settings-section">
            <Typography variant="h6" className="settings-section-title">
              Appearance
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <DarkModeIcon />
                </ListItemIcon>
                <ListItemText primary="Dark Mode" />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.appearance.darkMode}
                    onChange={(e) => handleSettingChange('appearance', 'darkMode', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Font Size" 
                  secondary={`${settings.appearance.fontSize}px`}
                />
                <ListItemSecondaryAction className="slider-action">
                  <Slider
                    value={settings.appearance.fontSize}
                    min={12}
                    max={24}
                    step={1}
                    onChange={(e, value) => handleSettingChange('appearance', 'fontSize', value)}
                    className="settings-slider"
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText primary="Message Spacing" />
                <ListItemSecondaryAction>
                  <FormControl variant="outlined" size="small" className="select-action">
                    <Select
                      value={settings.appearance.messageSpacing}
                      onChange={(e) => handleSettingChange('appearance', 'messageSpacing', e.target.value)}
                    >
                      <MenuItem value="compact">Compact</MenuItem>
                      <MenuItem value="normal">Normal</MenuItem>
                      <MenuItem value="relaxed">Relaxed</MenuItem>
                    </Select>
                  </FormControl>
                </ListItemSecondaryAction>
              </ListItem>
            </List>
            
            <Divider className="settings-divider" />
            
            <Typography variant="h6" className="settings-section-title">
              Language
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <LanguageIcon />
                </ListItemIcon>
                <ListItemText primary="Interface Language" />
                <ListItemSecondaryAction>
                  <FormControl variant="outlined" size="small" className="select-action">
                    <Select
                      value={settings.language}
                      onChange={(e) => handleDirectSettingChange('language', e.target.value)}
                    >
                      <MenuItem value="en">English</MenuItem>
                      <MenuItem value="es">Español</MenuItem>
                      <MenuItem value="fr">Français</MenuItem>
                      <MenuItem value="de">Deutsch</MenuItem>
                      <MenuItem value="zh">中文</MenuItem>
                      <MenuItem value="ja">日本語</MenuItem>
                    </Select>
                  </FormControl>
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </div>
        );
        
      case 1: // AI Models
        return (
          <div className="settings-section">
            <Typography variant="h6" className="settings-section-title">
              Model Settings
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <SmartToyIcon />
                </ListItemIcon>
                <ListItemText primary="Default Model" />
                <ListItemSecondaryAction>
                  <FormControl variant="outlined" size="small" className="select-action">
                    <Select
                      value={settings.defaultModel}
                      onChange={(e) => handleDirectSettingChange('defaultModel', e.target.value)}
                    >
                      <MenuItem value="digitalogy">Digitalogy LLM</MenuItem>
                      <MenuItem value="gpt-4">GPT-4</MenuItem>
                      <MenuItem value="claude-2">Claude 2</MenuItem>
                      <MenuItem value="llama-2">Llama 2</MenuItem>
                    </Select>
                  </FormControl>
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <DataUsageIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Temperature" 
                  secondary={`${settings.temperature} (${settings.temperature < 0.3 ? 'More focused' : settings.temperature > 0.7 ? 'More creative' : 'Balanced'})`}
                />
                <ListItemSecondaryAction className="slider-action">
                  <Slider
                    value={settings.temperature}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={(e, value) => handleDirectSettingChange('temperature', value)}
                    className="settings-slider"
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <StorageIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Max Tokens" 
                  secondary={`${settings.maxTokens} tokens`}
                />
                <ListItemSecondaryAction className="slider-action">
                  <Slider
                    value={settings.maxTokens}
                    min={256}
                    max={4096}
                    step={256}
                    onChange={(e, value) => handleDirectSettingChange('maxTokens', value)}
                    className="settings-slider"
                  />
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </div>
        );
        
      case 2: // Notifications
        return (
          <div className="settings-section">
            <Typography variant="h6" className="settings-section-title">
              Notification Settings
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <NotificationsIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Email Notifications" 
                  secondary="Receive important updates via email"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.notifications.emailNotifications}
                    onChange={(e) => handleSettingChange('notifications', 'emailNotifications', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <NotificationsIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Chat Completions" 
                  secondary="Get notified when AI completes a response"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.notifications.chatCompletions}
                    onChange={(e) => handleSettingChange('notifications', 'chatCompletions', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <NotificationsIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Marketing Emails" 
                  secondary="Receive promotional content and offers"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.notifications.marketingEmails}
                    onChange={(e) => handleSettingChange('notifications', 'marketingEmails', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </div>
        );
        
      case 3: // Privacy & Security
        return (
          <div className="settings-section">
            <Typography variant="h6" className="settings-section-title">
              Privacy Settings
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <SecurityIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Save Chat History" 
                  secondary="Store your conversations for future reference"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.privacy.saveHistory}
                    onChange={(e) => handleSettingChange('privacy', 'saveHistory', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <SecurityIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Share Usage Data" 
                  secondary="Help improve the AI by sharing anonymous usage data"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.privacy.shareUsageData}
                    onChange={(e) => handleSettingChange('privacy', 'shareUsageData', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <SecurityIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Allow Analytics" 
                  secondary="Enable analytics to improve your experience"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.privacy.allowAnalytics}
                    onChange={(e) => handleSettingChange('privacy', 'allowAnalytics', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
            </List>
            
            <Divider className="settings-divider" />
            
            <Typography variant="h6" className="settings-section-title danger-zone">
              Danger Zone
            </Typography>
            
            <div className="danger-zone-content">
              <Typography variant="body2" color="error" paragraph>
                Deleting your account will permanently remove all your data, including chat history, uploaded files, and personal information. This action cannot be undone.
              </Typography>
              
              {deleteDialogOpen ? (
                <div className="delete-confirmation">
                  <Typography variant="body2" paragraph>
                    To confirm deletion, type "DELETE" in the field below:
                  </Typography>
                  
                  <TextField
                    variant="outlined"
                    size="small"
                    fullWidth
                    value={deleteConfirmText}
                    onChange={(e) => setDeleteConfirmText(e.target.value)}
                    margin="normal"
                  />
                  
                  <div className="delete-actions">
                    <Button 
                      variant="outlined" 
                      onClick={() => setDeleteDialogOpen(false)}
                    >
                      Cancel
                    </Button>
                    
                    <Button 
                      variant="contained" 
                      color="error" 
                      startIcon={<DeleteIcon />}
                      onClick={handleDeleteAccount}
                    >
                      Delete Account
                    </Button>
                  </div>
                </div>
              ) : (
                <Button 
                  variant="outlined" 
                  color="error" 
                  startIcon={<DeleteIcon />}
                  onClick={() => setDeleteDialogOpen(true)}
                >
                  Delete Account
                </Button>
              )}
            </div>
          </div>
        );
        
      case 4: // Account
        return (
          <div className="settings-section">
            <Typography variant="h6" className="settings-section-title">
              Account Information
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Name" 
                  secondary={user?.name || 'Not set'}
                />
                <ListItemSecondaryAction>
                  <Button size="small" color="primary">
                    Edit
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Email" 
                  secondary={user?.email || 'Not set'}
                />
                <ListItemSecondaryAction>
                  <Button size="small" color="primary">
                    Change
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <VpnKeyIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Password" 
                  secondary="Last changed: 30 days ago"
                />
                <ListItemSecondaryAction>
                  <Button size="small" color="primary">
                    Change
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
            </List>
            
            <Divider className="settings-divider" />
            
            <Typography variant="h6" className="settings-section-title">
              Linked Accounts
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <img 
                    src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" 
                    alt="Google" 
                    width="24" 
                    height="24" 
                  />
                </ListItemIcon>
                <ListItemText 
                  primary="Google" 
                  secondary={user?.googleConnected ? 'Connected' : 'Not connected'}
                />
                <ListItemSecondaryAction>
                  <Button 
                    size="small" 
                    color={user?.googleConnected ? "error" : "primary"}
                  >
                    {user?.googleConnected ? 'Disconnect' : 'Connect'}
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <img 
                    src="https://upload.wikimedia.org/wikipedia/commons/c/c2/GitHub_Invertocat_Logo.svg" 
                    alt="GitHub" 
                    width="24" 
                    height="24" 
                  />
                </ListItemIcon>
                <ListItemText 
                  primary="GitHub" 
                  secondary={user?.githubConnected ? 'Connected' : 'Not connected'}
                />
                <ListItemSecondaryAction>
                  <Button 
                    size="small" 
                    color={user?.githubConnected ? "error" : "primary"}
                  >
                    {user?.githubConnected ? 'Disconnect' : 'Connect'}
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </div>
        );
        
      default:
        return null;
    }
  };

  return (
    <div className={`settings-container ${darkMode ? 'dark-mode' : ''}`}>
      <Paper elevation={0} className="settings-header">
        <Typography variant="h4" className="settings-title">
          Settings
        </Typography>
        
        <Button
          variant="contained"
          color="primary"
          startIcon={<SaveIcon />}
          onClick={handleSaveSettings}
          className="save-settings-button"
        >
          Save Changes
        </Button>
      </Paper>
      
      <div className="settings-content">
        <Paper className="settings-sidebar">
          <Tabs
            orientation="vertical"
            variant="scrollable"
            value={activeTab}
            onChange={handleTabChange}
            className="settings-tabs"
          >
            <Tab icon={<SettingsIcon />} label="General" />
            <Tab icon={<SmartToyIcon />} label="AI Models" />
            <Tab icon={<NotificationsIcon />} label="Notifications" />
            <Tab icon={<SecurityIcon />} label="Privacy & Security" />
            <Tab icon={<PersonIcon />} label="Account" />
          </Tabs>
        </Paper>
        
        <Paper className="settings-main">
          {renderTabContent()}
        </Paper>
      </div>
      
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

export default Settings; 