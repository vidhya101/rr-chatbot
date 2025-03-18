import api from './apiService';

// Register a new user
export const register = async (userData) => {
  try {
    const response = await api.post('/auth/register', userData);
    
    // Store token in localStorage
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
    }
    
    return response.data.user;
  } catch (error) {
    console.error('Registration error:', error);
    throw error.response?.data || error;
  }
};

// Login user
export const login = async (email, password) => {
  try {
    const response = await api.post('/auth/login', { email, password });
    
    // Store token in localStorage
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
    }
    
    return response.data.user;
  } catch (error) {
    console.error('Login error:', error);
    throw error.response?.data || error;
  }
};

// Login with social provider
export const loginWithProvider = async (provider) => {
  try {
    // In a real app, this would redirect to the provider's OAuth flow
    // For now, we'll simulate it with a direct API call
    const response = await api.post(`/auth/${provider}`);
    
    // Store token in localStorage
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
    }
    
    return response.data.user;
  } catch (error) {
    console.error(`${provider} login error:`, error);
    throw error.response?.data || error;
  }
};

// Logout user
export const logout = () => {
  try {
    // Remove token from localStorage
    localStorage.removeItem('token');
    
    // In a real app, you might also want to invalidate the token on the server
    // api.post('/auth/logout');
    
    return true;
  } catch (error) {
    console.error('Logout error:', error);
    return false;
  }
};

// Check if user is authenticated
export const isAuthenticated = () => {
  return !!localStorage.getItem('token');
};

// Get current user
export const getCurrentUser = async () => {
  try {
    if (!isAuthenticated()) {
      return null;
    }
    
    const response = await api.get('/auth/me');
    return response.data;
  } catch (error) {
    console.error('Get current user error:', error);
    return null;
  }
};

// Request password reset
export const requestPasswordReset = async (email) => {
  try {
    const response = await api.post('/auth/forgot-password', { email });
    return response.data;
  } catch (error) {
    console.error('Password reset request error:', error);
    throw error.response?.data || error;
  }
};

// Reset password
export const resetPassword = async (token, newPassword) => {
  try {
    const response = await api.post('/auth/reset-password', { 
      token, 
      password: newPassword 
    });
    return response.data;
  } catch (error) {
    console.error('Password reset error:', error);
    throw error.response?.data || error;
  }
};

// Change password
export const changePassword = async (currentPassword, newPassword) => {
  try {
    const response = await api.post('/auth/change-password', {
      currentPassword,
      newPassword
    });
    return response.data;
  } catch (error) {
    console.error('Change password error:', error);
    throw error.response?.data || error;
  }
};

// Verify email
export const verifyEmail = async (token) => {
  try {
    const response = await api.post('/auth/verify-email', { token });
    return response.data;
  } catch (error) {
    console.error('Email verification error:', error);
    throw error.response?.data || error;
  }
};

// Create a named export object
const authService = {
  register,
  login,
  loginWithProvider,
  logout,
  isAuthenticated,
  getCurrentUser,
  requestPasswordReset,
  resetPassword,
  changePassword,
  verifyEmail
};

export default authService; 