import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';

// Components
import Dashboard from './components/Dashboard';
import MonitoringDashboard from './components/MonitoringDashboard';
import ChatInterface from './components/ChatInterface';
import DataModeling from './components/DataModeling';
import HistoryView from './components/HistoryView';
import UploadView from './components/UploadView';
import Visualizations from './components/Visualizations';
import RawDataProcessor from './components/RawDataProcessor';

// Contexts
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import theme from './theme';

const App = () => {
  const [activeModel, setActiveModel] = useState('gpt-3.5-turbo');
  const [darkMode, setDarkMode] = useState(false);

  return (
    <Router>
      <CustomThemeProvider>
        <AuthProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <Layout>
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
                <Route path="/data-modeling" element={<DataModeling />} />
                <Route path="/raw-data" element={<RawDataProcessor />} />
                <Route path="/history" element={<HistoryView />} />
                <Route path="/upload" element={<UploadView />} />
                <Route path="/visualizations" element={<Visualizations />} />
              </Routes>
            </Layout>
          </ThemeProvider>
        </AuthProvider>
      </CustomThemeProvider>
    </Router>
  );
};

export default App; 