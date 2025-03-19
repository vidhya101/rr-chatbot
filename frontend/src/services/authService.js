import api from './apiService';

// Constants
const TOKEN_KEY = 'token';
const USER_KEY = 'user';
const TOKEN_EXPIRY_KEY = 'tokenExpiry';

// Helper functions
const setToken = (token) => {
  localStorage.setItem(TOKEN_KEY, token);
  // Set token expiry to 24 hours from now
  const expiry = new Date().getTime() + 24 * 60 * 60 * 1000;
  localStorage.setItem(TOKEN_EXPIRY_KEY, expiry.toString());
};

const clearAuthData = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_EXPIRY_KEY);
};

const isTokenExpired = () => {
  const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
  if (!expiry) return true;
  return new Date().getTime() > parseInt(expiry);
};

// Register a new user
export const register = async (userData) => {
  try {
    const response = await api.post('/auth/register', userData);
    
    if (response.data.token) {
      setToken(response.data.token);
      localStorage.setItem(USER_KEY, JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.message || 'Registration failed';
    throw new Error(errorMessage);
  }
};

// Login user
export const login = async (email, password) => {
  try {
    const response = await api.post('/auth/login', { email, password });
    
    if (response.data.token) {
      setToken(response.data.token);
      localStorage.setItem(USER_KEY, JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.message || 'Login failed';
    throw new Error(errorMessage);
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
export const logout = async () => {
  try {
    // Call logout endpoint to invalidate token on server
    await api.post('/auth/logout');
    clearAuthData();
    return true;
  } catch (error) {
    console.error('Logout error:', error);
    // Still clear local data even if server call fails
    clearAuthData();
    return true;
  }
};

// Check if user is authenticated
export const isAuthenticated = () => {
  const token = localStorage.getItem(TOKEN_KEY);
  return !!token && !isTokenExpired();
};

// Get current user
export const getCurrentUser = async () => {
  try {
    if (!isAuthenticated()) {
      return null;
    }
    
    const response = await api.get('/auth/me');
    const userData = response.data;
    localStorage.setItem(USER_KEY, JSON.stringify(userData));
    return userData;
  } catch (error) {
    console.error('Get current user error:', error);
    // If token is invalid, clear auth data
    if (error.response?.status === 401) {
      clearAuthData();
    }
    return null;
  }
};

// Request password reset
export const requestPasswordReset = async (email) => {
  try {
    const response = await api.post('/auth/forgot-password', { email });
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.message || 'Password reset request failed';
    throw new Error(errorMessage);
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
    const errorMessage = error.response?.data?.message || 'Password reset failed';
    throw new Error(errorMessage);
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
    const errorMessage = error.response?.data?.message || 'Password change failed';
    throw new Error(errorMessage);
  }
};

// Verify email
export const verifyEmail = async (token) => {
  try {
    const response = await api.post('/auth/verify-email', { token });
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.message || 'Email verification failed';
    throw new Error(errorMessage);
  }
};

// Refresh token
export const refreshToken = async () => {
  try {
    const response = await api.post('/auth/refresh-token');
    if (response.data.token) {
      setToken(response.data.token);
    }
    return response.data;
  } catch (error) {
    console.error('Token refresh error:', error);
    clearAuthData();
    throw new Error('Session expired. Please login again.');
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
  verifyEmail,
  refreshToken
};

export default authService; 