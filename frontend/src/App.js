import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Material UI
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';
import Register from './components/Register';
import Dashboard from './components/Dashboard';
import Settings from './components/Settings';
import Profile from './components/Profile';
import FileUpload from './components/FileUpload';
import NotFound from './components/NotFound';
import AdminDashboard from './components/AdminDashboard';
import DataAnalysis from './components/DataAnalysis';

// Services
import { isAuthenticated, getCurrentUser, logout } from './services/authService';
import { getChatHistory } from './services/userService';
import { listModels, switchModel } from './services/apiService';

// Protected Route component
const ProtectedRoute = ({ children }) => {
  return isAuthenticated() ? children : <Navigate to="/login" />;
};

function App() {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('darkMode') === 'true');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [user, setUser] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [activeModel, setActiveModel] = useState('digitalogy');

  // Create theme based on dark mode preference
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#4a90e2',
      },
      secondary: {
        main: '#f50057',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarWidth: 'thin',
            '&::-webkit-scrollbar': {
              width: '8px',
              height: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: darkMode ? '#333' : '#f1f1f1',
            },
            '&::-webkit-scrollbar-thumb': {
              background: darkMode ? '#888' : '#ccc',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb:hover': {
              background: darkMode ? '#555' : '#999',
            },
          },
        },
      },
    },
  });

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode.toString());
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Handle login success
  const handleLoginSuccess = (userData) => {
    setUser(userData);
  };

  // Handle register success
  const handleRegisterSuccess = (userData) => {
    setUser(userData);
  };

  // Handle logout
  const handleLogout = () => {
    logout();
    setUser(null);
  };

  // Handle model switch
  const handleSwitchModel = async (modelName) => {
    try {
      await switchModel(modelName);
      setActiveModel(modelName);
    } catch (error) {
      console.error('Error switching model:', error);
    }
  };

  // Load user data and chat history on mount
  useEffect(() => {
    const loadUserData = async () => {
      try {
        if (isAuthenticated()) {
          // Load user data
          const userData = await getCurrentUser();
          setUser(userData);
          
          // Load chat history
          const history = await getChatHistory();
          setChatHistory(history || []);
          
          // Load active model
          const models = await listModels();
          if (models && models.length > 0) {
            const defaultModel = models.find(model => model.isDefault);
            if (defaultModel) {
              setActiveModel(defaultModel.name);
            }
          }
        }
      } catch (error) {
        console.error('Error loading user data:', error);
      }
    };
    
    loadUserData();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="app">
          <Header 
            toggleSidebar={toggleSidebar} 
            darkMode={darkMode} 
            toggleDarkMode={toggleDarkMode} 
            user={user}
            handleLogout={handleLogout}
            activeModel={activeModel}
          />
          
          <div className="app-container">
            <Sidebar 
              isOpen={sidebarOpen} 
              chatHistory={chatHistory} 
              activeModel={activeModel}
              switchModel={handleSwitchModel}
              darkMode={darkMode}
            />
            
            <main className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}>
              <Routes>
                <Route 
                  path="/" 
                  element={
                    <ChatInterface 
                      activeModel={activeModel} 
                      darkMode={darkMode} 
                    />
                  } 
                />
                
                <Route 
                  path="/login" 
                  element={
                    isAuthenticated() ? (
                      <Navigate to="/" />
                    ) : (
                      <Login 
                        darkMode={darkMode} 
                        onLoginSuccess={handleLoginSuccess} 
                      />
                    )
                  } 
                />
                
                <Route 
                  path="/register" 
                  element={
                    isAuthenticated() ? (
                      <Navigate to="/" />
                    ) : (
                      <Register 
                        darkMode={darkMode} 
                        onRegisterSuccess={handleRegisterSuccess} 
                      />
                    )
                  } 
                />
                
                <Route 
                  path="/dashboard" 
                  element={
                    <ProtectedRoute>
                      <Dashboard darkMode={darkMode} />
                    </ProtectedRoute>
                  } 
                />
                
                <Route 
                  path="/admin" 
                  element={
                    <ProtectedRoute>
                      <AdminDashboard darkMode={darkMode} />
                    </ProtectedRoute>
                  } 
                />
                
                <Route 
                  path="/files" 
                  element={
                    <ProtectedRoute>
                      <FileUpload darkMode={darkMode} />
                    </ProtectedRoute>
                  } 
                />
                
                <Route 
                  path="/data-analysis" 
                  element={
                    <DataAnalysis darkMode={darkMode} />
                  } 
                />
                
                <Route 
                  path="/settings" 
                  element={
                    <ProtectedRoute>
                      <Settings 
                        darkMode={darkMode} 
                        toggleDarkMode={toggleDarkMode} 
                        user={user} 
                      />
                    </ProtectedRoute>
                  } 
                />
                
                <Route 
                  path="/profile" 
                  element={
                    <ProtectedRoute>
                      <Profile darkMode={darkMode} user={user} />
                    </ProtectedRoute>
                  } 
                />
                
                <Route path="*" element={<NotFound darkMode={darkMode} />} />
              </Routes>
            </main>
          </div>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App; 