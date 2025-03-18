import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Header.css';

// Material UI components
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton, 
  Button, 
  Avatar, 
  Menu, 
  MenuItem, 
  ListItemIcon,
  Divider,
  Tooltip
} from '@mui/material';

// Material UI Icons
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpIcon from '@mui/icons-material/Help';
import LogoutIcon from '@mui/icons-material/Logout';
import PersonIcon from '@mui/icons-material/Person';
import SupervisorAccountIcon from '@mui/icons-material/SupervisorAccount';
import BarChartIcon from '@mui/icons-material/BarChart';
import AnalyticsIcon from '@mui/icons-material/Analytics';

const Header = ({ 
  toggleSidebar, 
  darkMode, 
  toggleDarkMode, 
  user, 
  handleLogout,
  activeModel
}) => {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogoutClick = () => {
    handleClose();
    handleLogout();
  };

  return (
    <AppBar 
      position="static" 
      color="default" 
      className={`header ${darkMode ? 'dark-mode' : ''}`}
    >
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={toggleSidebar}
          className="menu-button"
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant="h6" className="logo">
          <Link to="/">AI Chatbot</Link>
        </Typography>
        
        {activeModel && (
          <div className="active-model">
            <Typography variant="body2">
              Model: <span className="model-name">{activeModel}</span>
            </Typography>
          </div>
        )}
        
        <div className="header-spacer"></div>
        
        <Tooltip title="Data Analysis">
          <Button
            color="inherit"
            component={Link}
            to="/data-analysis"
            className="nav-button"
            startIcon={<AnalyticsIcon />}
          >
            Data Analysis
          </Button>
        </Tooltip>
        
        <Tooltip title={`Switch to ${darkMode ? 'Light' : 'Dark'} Mode`}>
          <IconButton 
            color="inherit" 
            onClick={toggleDarkMode}
            className="theme-toggle"
          >
            {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
        </Tooltip>
        
        {user ? (
          <>
            <Tooltip title="Account">
              <IconButton
                onClick={handleMenu}
                color="inherit"
                className="account-button"
              >
                {user.avatar ? (
                  <Avatar 
                    src={user.avatar} 
                    alt={user.name || user.email} 
                    className="user-avatar"
                  />
                ) : (
                  <AccountCircleIcon />
                )}
              </IconButton>
            </Tooltip>
            
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={open}
              onClose={handleClose}
              className="user-menu"
            >
              <div className="user-info">
                <Typography variant="subtitle1">{user.name || 'User'}</Typography>
                <Typography variant="body2" color="textSecondary">{user.email}</Typography>
              </div>
              <Divider />
              <MenuItem component={Link} to="/profile" onClick={handleClose}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" className="menu-icon" />
                </ListItemIcon>
                Profile
              </MenuItem>
              <MenuItem component={Link} to="/settings" onClick={handleClose}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" className="menu-icon" />
                </ListItemIcon>
                Settings
              </MenuItem>
              {user.role === 'admin' && (
                <MenuItem component={Link} to="/admin" onClick={handleClose}>
                  <ListItemIcon>
                    <SupervisorAccountIcon fontSize="small" className="menu-icon" />
                  </ListItemIcon>
                  Admin Dashboard
                </MenuItem>
              )}
              <Divider />
              <MenuItem onClick={handleLogoutClick}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" className="menu-icon" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </>
        ) : (
          <div className="auth-buttons">
            <Button 
              color="inherit" 
              component={Link} 
              to="/login"
              className="login-button"
            >
              Login
            </Button>
            <Button 
              variant="contained" 
              color="primary" 
              component={Link} 
              to="/register"
              className="register-button"
            >
              Sign Up
            </Button>
          </div>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Header; 