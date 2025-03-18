import axios from 'axios';
import { getAccessToken, getRefreshToken, setAuth, clearAuth, isTokenExpired } from './auth';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  // Add timeout
  timeout: 10000,
  // Remove withCredentials since we're not using cookies
  withCredentials: false
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = getAccessToken();
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Log request
    console.log(`Making ${config.method.toUpperCase()} request to ${config.url}`, config.data);
    
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Log successful response
    console.log(`Response from ${response.config.url}:`, response.data);
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Log error response
    console.error('Response error:', {
      url: originalRequest?.url,
      status: error.response?.status,
      data: error.response?.data,
      error: error.message
    });
    
    // If error is 401 Unauthorized and not a retry
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      // Try to refresh token
      const refreshToken = getRefreshToken();
      
      if (refreshToken && !isTokenExpired(refreshToken)) {
        try {
          const response = await axios.post(
            `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/auth/refresh`,
            {},
            {
              headers: {
                'Authorization': `Bearer ${refreshToken}`
              }
            }
          );
          
          // Update tokens
          setAuth({
            access_token: response.data.access_token
          });
          
          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${response.data.access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
          clearAuth();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        clearAuth();
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

/**
 * Make a GET request
 * @param {string} url - API endpoint
 * @param {Object} params - Query parameters
 * @param {Object} options - Additional options
 * @returns {Promise} Axios promise
 */
export const get = (url, params = {}, options = {}) => {
  return api.get(url, { params, ...options });
};

/**
 * Make a POST request
 * @param {string} url - API endpoint
 * @param {Object} data - Request body
 * @param {Object} options - Additional options
 * @returns {Promise} Axios promise
 */
export const post = (url, data = {}, options = {}) => {
  return api.post(url, data, options);
};

/**
 * Make a PUT request
 * @param {string} url - API endpoint
 * @param {Object} data - Request body
 * @param {Object} options - Additional options
 * @returns {Promise} Axios promise
 */
export const put = (url, data = {}, options = {}) => {
  return api.put(url, data, options);
};

/**
 * Make a DELETE request
 * @param {string} url - API endpoint
 * @param {Object} options - Additional options
 * @returns {Promise} Axios promise
 */
export const del = (url, options = {}) => {
  return api.delete(url, options);
};

/**
 * Upload a file
 * @param {string} url - API endpoint
 * @param {File} file - File to upload
 * @param {Object} additionalData - Additional form data
 * @param {Function} onProgress - Progress callback
 * @returns {Promise} Axios promise
 */
export const uploadFile = (url, file, additionalData = {}, onProgress = null) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add additional data to form data
  Object.keys(additionalData).forEach(key => {
    formData.append(key, additionalData[key]);
  });
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  };
  
  // Add progress event listener if provided
  if (onProgress) {
    config.onUploadProgress = (progressEvent) => {
      const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      onProgress(percentCompleted);
    };
  }
  
  return api.post(url, formData, config);
};

export default api; 