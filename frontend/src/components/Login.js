import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Login.css';

// Material UI components
import {
  Paper,
  Typography,
  TextField,
  Button,
  Divider,
  IconButton,
  InputAdornment,
  Alert,
  CircularProgress
} from '@mui/material';

// Material UI Icons
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import GoogleIcon from '@mui/icons-material/Google';
import GitHubIcon from '@mui/icons-material/GitHub';
import TwitterIcon from '@mui/icons-material/Twitter';

// Services
import { login, loginWithProvider } from '../services/authService';

const Login = ({ darkMode, onLoginSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const navigate = useNavigate();

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const userData = await login(email, password);
      
      // Call the onLoginSuccess callback with user data
      onLoginSuccess(userData);
      
      // Redirect to home page
      navigate('/');
    } catch (err) {
      console.error('Login error:', err);
      setError(err.message || 'Failed to login. Please check your credentials and try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle social login
  const handleSocialLogin = async (provider) => {
    setLoading(true);
    setError(null);
    
    try {
      const userData = await loginWithProvider(provider);
      
      // Call the onLoginSuccess callback with user data
      onLoginSuccess(userData);
      
      // Redirect to home page
      navigate('/');
    } catch (err) {
      console.error(`${provider} login error:`, err);
      setError(`Failed to login with ${provider}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className={`login-container ${darkMode ? 'dark-mode' : ''}`}>
      <Paper elevation={3} className="login-paper">
        <Typography variant="h4" className="login-title">
          Welcome Back
        </Typography>
        
        <Typography variant="body2" color="textSecondary" className="login-subtitle">
          Sign in to continue to AI Chatbot
        </Typography>
        
        {error && (
          <Alert severity="error" className="login-alert">
            {error}
          </Alert>
        )}
        
        <form onSubmit={handleSubmit} className="login-form">
          <TextField
            label="Email"
            type="email"
            fullWidth
            variant="outlined"
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={loading}
            required
          />
          
          <TextField
            label="Password"
            type={showPassword ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            required
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={togglePasswordVisibility}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
          
          <div className="login-options">
            <Link to="/forgot-password" className="forgot-password-link">
              Forgot Password?
            </Link>
          </div>
          
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            size="large"
            className="login-button"
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Sign In'}
          </Button>
        </form>
        
        <Divider className="login-divider">
          <Typography variant="body2" color="textSecondary">
            OR
          </Typography>
        </Divider>
        
        <div className="social-login">
          <Button
            variant="outlined"
            startIcon={<GoogleIcon />}
            onClick={() => handleSocialLogin('google')}
            disabled={loading}
            className="social-button google"
          >
            Google
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<GitHubIcon />}
            onClick={() => handleSocialLogin('github')}
            disabled={loading}
            className="social-button github"
          >
            GitHub
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<TwitterIcon />}
            onClick={() => handleSocialLogin('twitter')}
            disabled={loading}
            className="social-button twitter"
          >
            Twitter
          </Button>
        </div>
        
        <div className="register-prompt">
          <Typography variant="body2">
            Don't have an account?{' '}
            <Link to="/register" className="register-link">
              Sign Up
            </Link>
          </Typography>
        </div>
      </Paper>
    </div>
  );
};

export default Login; 