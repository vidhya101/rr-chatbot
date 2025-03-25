import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, AppBar, Toolbar, Drawer, List, ListItem, ListItemIcon, ListItemText, Typography, IconButton } from '@mui/material';
import { Link } from 'react-router-dom';
import { 
  Dashboard as DashboardIcon, 
  Monitor as MonitorIcon,
  Chat as ChatIcon,
  Menu as MenuIcon,
  ModelTraining as ModelTrainingIcon,
  History as HistoryIcon,
  Upload as UploadIcon,
  BarChart as VisualizationIcon
} from '@mui/icons-material';

// Components
import Dashboard from './components/Dashboard';
import MonitoringDashboard from './components/MonitoringDashboard';
import ChatInterface from './components/ChatInterface';
import DataModeling from './components/DataModeling';
import HistoryView from './components/HistoryView';
import UploadView from './components/UploadView';
import VisualizationView from './components/VisualizationView';

// Contexts
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import theme from './theme';

const App = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeModel, setActiveModel] = useState('gpt-3.5-turbo');
  const [darkMode, setDarkMode] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Monitoring', icon: <MonitorIcon />, path: '/monitoring' },
    { text: 'Chat', icon: <ChatIcon />, path: '/chat' },
    { text: 'Data Modeling', icon: <ModelTrainingIcon />, path: '/modeling' },
    { text: 'History', icon: <HistoryIcon />, path: '/history' },
    { text: 'Upload', icon: <UploadIcon />, path: '/upload' },
    { text: 'Visualizations', icon: <VisualizationIcon />, path: '/visualizations' }
  ];

  const drawer = (
    <div>
      <Toolbar />
      <List>
        {menuItems.map((item) => (
          <ListItem button key={item.text} component={Link} to={item.path}>
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Router>
      <CustomThemeProvider>
        <AuthProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ display: 'flex' }}>
              <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                <Toolbar>
                  <IconButton
                    color="inherit"
                    aria-label="open drawer"
                    edge="start"
                    onClick={handleDrawerToggle}
                    sx={{ mr: 2, display: { sm: 'none' } }}
                  >
                    <MenuIcon />
                  </IconButton>
                  <Typography variant="h6" noWrap component="div">
                    RR Chatbot
                  </Typography>
                </Toolbar>
              </AppBar>
              <Drawer
                variant="permanent"
                sx={{
                  width: 240,
                  flexShrink: 0,
                  '& .MuiDrawer-paper': {
                    width: 240,
                    boxSizing: 'border-box',
                  },
                }}
              >
                {drawer}
              </Drawer>
              <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
                <Toolbar />
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/monitoring" element={<MonitoringDashboard />} />
                  <Route 
                    path="/chat" 
                    element={
                      <ChatInterface 
                        activeModel={activeModel}
                        darkMode={darkMode}
                        onModelChange={setActiveModel}
                      />
                    } 
                  />
                  <Route path="/modeling" element={<DataModeling />} />
                  <Route path="/history" element={<HistoryView />} />
                  <Route path="/upload" element={<UploadView />} />
                  <Route path="/visualizations" element={<VisualizationView />} />
                </Routes>
              </Box>
            </Box>
          </ThemeProvider>
        </AuthProvider>
      </CustomThemeProvider>
    </Router>
  );
};

export default App; 