import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Register.css';

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
  CircularProgress,
  Checkbox,
  FormControlLabel
} from '@mui/material';

// Material UI Icons
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import GoogleIcon from '@mui/icons-material/Google';
import GitHubIcon from '@mui/icons-material/GitHub';
import TwitterIcon from '@mui/icons-material/Twitter';

// Services
import { register, loginWithProvider } from '../services/authService';

const Register = ({ darkMode, onRegisterSuccess }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [agreeToTerms, setAgreeToTerms] = useState(false);
  
  const navigate = useNavigate();

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!formData.name || !formData.email || !formData.password || !formData.confirmPassword) {
      setError('All fields are required.');
      return;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    
    if (!agreeToTerms) {
      setError('You must agree to the Terms of Service and Privacy Policy.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const userData = await register(formData);
      
      // Call the onRegisterSuccess callback with user data
      onRegisterSuccess(userData);
      
      // Redirect to home page
      navigate('/');
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.message || 'Failed to register. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle social login/register
  const handleSocialRegister = async (provider) => {
    setLoading(true);
    setError(null);
    
    try {
      const userData = await loginWithProvider(provider);
      
      // Call the onRegisterSuccess callback with user data
      onRegisterSuccess(userData);
      
      // Redirect to home page
      navigate('/');
    } catch (err) {
      console.error(`${provider} registration error:`, err);
      setError(`Failed to register with ${provider}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className={`register-container ${darkMode ? 'dark-mode' : ''}`}>
      <Paper elevation={3} className="register-paper">
        <Typography variant="h4" className="register-title">
          Create Account
        </Typography>
        
        <Typography variant="body2" color="textSecondary" className="register-subtitle">
          Sign up to start using AI Chatbot
        </Typography>
        
        {error && (
          <Alert severity="error" className="register-alert">
            {error}
          </Alert>
        )}
        
        <form onSubmit={handleSubmit} className="register-form">
          <TextField
            label="Full Name"
            name="name"
            fullWidth
            variant="outlined"
            margin="normal"
            value={formData.name}
            onChange={handleChange}
            disabled={loading}
            required
          />
          
          <TextField
            label="Email"
            name="email"
            type="email"
            fullWidth
            variant="outlined"
            margin="normal"
            value={formData.email}
            onChange={handleChange}
            disabled={loading}
            required
          />
          
          <TextField
            label="Password"
            name="password"
            type={showPassword ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            margin="normal"
            value={formData.password}
            onChange={handleChange}
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
          
          <TextField
            label="Confirm Password"
            name="confirmPassword"
            type={showPassword ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            margin="normal"
            value={formData.confirmPassword}
            onChange={handleChange}
            disabled={loading}
            required
          />
          
          <FormControlLabel
            control={
              <Checkbox
                checked={agreeToTerms}
                onChange={(e) => setAgreeToTerms(e.target.checked)}
                color="primary"
                disabled={loading}
              />
            }
            label={
              <Typography variant="body2">
                I agree to the{' '}
                <Link to="/terms" className="terms-link">
                  Terms of Service
                </Link>{' '}
                and{' '}
                <Link to="/privacy" className="terms-link">
                  Privacy Policy
                </Link>
              </Typography>
            }
            className="terms-checkbox"
          />
          
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            size="large"
            className="register-button"
            disabled={loading || !agreeToTerms}
          >
            {loading ? <CircularProgress size={24} /> : 'Sign Up'}
          </Button>
        </form>
        
        <Divider className="register-divider">
          <Typography variant="body2" color="textSecondary">
            OR
          </Typography>
        </Divider>
        
        <div className="social-register">
          <Button
            variant="outlined"
            startIcon={<GoogleIcon />}
            onClick={() => handleSocialRegister('google')}
            disabled={loading}
            className="social-button google"
          >
            Continue with Google
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<GitHubIcon />}
            onClick={() => handleSocialRegister('github')}
            disabled={loading}
            className="social-button github"
          >
            Continue with GitHub
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<TwitterIcon />}
            onClick={() => handleSocialRegister('twitter')}
            disabled={loading}
            className="social-button twitter"
          >
            Continue with Twitter
          </Button>
        </div>
        
        <div className="login-prompt">
          <Typography variant="body2">
            Already have an account?{' '}
            <Link to="/login" className="login-link">
              Sign In
            </Link>
          </Typography>
        </div>
      </Paper>
    </div>
  );
};

export default Register; 