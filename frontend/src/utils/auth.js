/**
 * Authentication utility functions
 */

// Token storage keys
const ACCESS_TOKEN_KEY = 'rr_chatbot_access_token';
const REFRESH_TOKEN_KEY = 'rr_chatbot_refresh_token';
const USER_KEY = 'rr_chatbot_user';

/**
 * Set authentication tokens and user data in local storage
 * @param {Object} data - Authentication data
 * @param {string} data.access_token - JWT access token
 * @param {string} data.refresh_token - JWT refresh token
 * @param {Object} data.user - User data
 */
export const setAuth = (data) => {
  if (data.access_token) {
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
  }
  
  if (data.refresh_token) {
    localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token);
  }
  
  if (data.user) {
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
  }
};

/**
 * Clear authentication data from local storage
 */
export const clearAuth = () => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

/**
 * Get access token from local storage
 * @returns {string|null} Access token or null if not found
 */
export const getAccessToken = () => {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
};

/**
 * Get refresh token from local storage
 * @returns {string|null} Refresh token or null if not found
 */
export const getRefreshToken = () => {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
};

/**
 * Get user data from local storage
 * @returns {Object|null} User data or null if not found
 */
export const getUser = () => {
  const userData = localStorage.getItem(USER_KEY);
  
  if (!userData) {
    return null;
  }
  
  try {
    return JSON.parse(userData);
  } catch (error) {
    console.error('Error parsing user data:', error);
    return null;
  }
};

/**
 * Update user data in local storage
 * @param {Object} userData - Updated user data
 */
export const updateUser = (userData) => {
  if (!userData) {
    return;
  }
  
  const currentUser = getUser();
  
  if (!currentUser) {
    localStorage.setItem(USER_KEY, JSON.stringify(userData));
    return;
  }
  
  localStorage.setItem(USER_KEY, JSON.stringify({
    ...currentUser,
    ...userData
  }));
};

/**
 * Check if user is authenticated
 * @returns {boolean} True if authenticated, false otherwise
 */
export const isAuthenticated = () => {
  return !!getAccessToken();
};

/**
 * Parse JWT token to get payload
 * @param {string} token - JWT token
 * @returns {Object|null} Token payload or null if invalid
 */
export const parseToken = (token) => {
  if (!token) {
    return null;
  }
  
  try {
    // Split the token and get the payload part
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error('Error parsing token:', error);
    return null;
  }
};

/**
 * Check if token is expired
 * @param {string} token - JWT token
 * @returns {boolean} True if expired, false otherwise
 */
export const isTokenExpired = (token) => {
  if (!token) {
    return true;
  }
  
  const payload = parseToken(token);
  
  if (!payload || !payload.exp) {
    return true;
  }
  
  // exp is in seconds, Date.now() is in milliseconds
  return payload.exp * 1000 < Date.now();
}; 