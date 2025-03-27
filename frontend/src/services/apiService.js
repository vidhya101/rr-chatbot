/**
 * API Service
 * 
 * This service handles communication with the backend API.
 */

import axios from 'axios';

// Create an axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 30000 // 30 seconds
});

// Add a request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle session expiration
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Chat API functions
export const sendMessage = async ({ message, chatHistory, files, model }) => {
  try {
    // First try the simple-chat endpoint
    try {
      const formData = new FormData();
      
      // Add message and model to form data
      const payload = {
        message,
        chatHistory,
        model: model || 'mistral'
      };
      
      formData.append('data', JSON.stringify(payload));
      
      // Add files if any
      if (files && files.length > 0) {
        files.forEach((file, index) => {
          formData.append(`file${index}`, file);
        });
      }
      
      const response = await api.post('/simple-chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout
      });
      
      return response.data;
    } catch (simpleError) {
      console.log('Simple chat endpoint failed, trying public chat endpoint', simpleError);
      
      // If simple-chat fails, try the public/chat endpoint
      const response = await api.post('/public/chat', {
        message,
        chatHistory,
        model: model || 'mistral'
      }, {
        timeout: 60000, // 60 second timeout
      });
      
      return response.data;
    }
  } catch (error) {
    console.error('Error sending message:', error);
    
    // Check if it's a timeout error
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      throw new Error('The request took too long to complete. The AI might be busy or unavailable at the moment.');
    }
    
    // Check if it's a network error
    if (error.message.includes('Network Error')) {
      throw new Error('Unable to connect to the server. Please check your internet connection and try again.');
    }
    
    // Handle API error responses
    if (error.response) {
      const status = error.response.status;
      const errorData = error.response.data;
      
      if (status === 401 || status === 403) {
        throw new Error('You need to be logged in to use this feature.');
      } else if (status === 404) {
        throw new Error('The chat service could not be found. Please try again later.');
      } else if (status === 429) {
        throw new Error('You\'ve sent too many messages. Please wait a moment and try again.');
      } else if (status >= 500) {
        throw new Error('The server encountered an error. Our team has been notified.');
      } else if (errorData && errorData.message) {
        throw new Error(errorData.message);
      }
    }
    
    // Default error message
    throw new Error('Failed to send message. Please try again.');
  }
};

export const stopGeneration = async () => {
  try {
    const response = await api.post('/chat/stop');
    return response.data;
  } catch (error) {
    console.error('Error stopping generation:', error);
    throw error;
  }
};

export const getChatHistory = async () => {
  try {
    const response = await api.get('/chat/history');
    return response.data;
  } catch (error) {
    console.error('Error fetching chat history:', error);
    throw error;
  }
};

// Model API functions
export const listModels = async () => {
  try {
    const response = await api.get('/models', {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    // Return a default set of models if the API call fails
    return {
      models: [
        { id: 'mistral', name: 'Mistral (Local)', isDefault: true, provider: 'ollama' },
        { id: 'mistral-small', name: 'Mistral Small (API)', isDefault: false, provider: 'mistral' }
      ],
      ollama_status: 'unknown'
    };
  }
};

export const switchModel = async (modelName) => {
  try {
    const response = await api.post('/chat', { model: modelName, message: "Switching to model: " + modelName });
    return response.data;
  } catch (error) {
    console.error('Error switching model:', error);
    throw error;
  }
};

// File API functions
export const uploadFile = async (formData, onUploadProgress) => {
  console.log('Starting file upload...');
  try {
    console.log('Uploading file to endpoint:', api.defaults.baseURL.replace('/api', '') + '/api/data/upload');
    
    // Check if formData contains a file
    const hasFile = formData.has('file');
    const file = formData.get('file');
    console.log('FormData contains file:', hasFile, file ? file.name : 'No file');
    
    // Use axios directly instead of the api instance to control the full URL
    const response = await axios.post('http://localhost:5000/api/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        if (onUploadProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log('Upload progress:', progress + '%');
          onUploadProgress(progress);
        }
      }
    });
    
    console.log('Upload successful, response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error uploading file:', error);
    if (error.response) {
      console.error('Server error response:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('No response received from server. Network error?');
    } else {
      console.error('Error setting up request:', error.message);
    }
    throw error;
  }
};

export const listFiles = async () => {
  try {
    const response = await api.get('/files');
    return response.data;
  } catch (error) {
    console.error('Error fetching files:', error);
    throw error;
  }
};

export const deleteFile = async (fileId) => {
  try {
    const response = await api.delete(`/files/${fileId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting file:', error);
    throw error;
  }
};

// Dashboard API functions
export const getDashboards = async (category = 'recent') => {
  try {
    const response = await api.get(`/dashboards?category=${category}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboards:', error);
    throw error;
  }
};

export const getDashboard = async (dashboardId) => {
  try {
    const response = await api.get(`/dashboards/${dashboardId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard:', error);
    throw error;
  }
};

export const createDashboard = async (dashboardData) => {
  try {
    const response = await api.post('/dashboards', dashboardData);
    return response.data;
  } catch (error) {
    console.error('Error creating dashboard:', error);
    throw error;
  }
};

export const updateDashboard = async (dashboardId, dashboardData) => {
  try {
    const response = await api.put(`/dashboards/${dashboardId}`, dashboardData);
    return response.data;
  } catch (error) {
    console.error('Error updating dashboard:', error);
    throw error;
  }
};

export const deleteDashboard = async (dashboardId) => {
  try {
    const response = await api.delete(`/dashboards/${dashboardId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting dashboard:', error);
    throw error;
  }
};

/**
 * Send a message with files to the chat API
 * @param {FormData} formData - FormData containing message data and files
 * @param {Object} settings - Model settings
 * @returns {Promise<Object>} - API response
 */
export const sendMessageWithFiles = async (formData, settings = {}) => {
  try {
    // First try the simple-chat endpoint
    try {
      const response = await fetch('/api/simple-chat', {
        method: 'POST',
        body: formData,
        timeout: 60000 // 60 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('Error using simple-chat endpoint, falling back to public chat:', error);
      
      // Fall back to public chat endpoint
      const response = await fetch('/api/public/chat', {
        method: 'POST',
        body: formData,
        timeout: 60000 // 60 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    }
  } catch (error) {
    console.error('Error sending message with files:', error);
    throw error;
  }
};

// Visualization API functions
export const generateDashboard = async (filePath) => {
  try {
    const response = await api.post('/visualization/dashboard', {
      file_path: filePath
    });
    return response.data;
  } catch (error) {
    console.error('Error generating dashboard:', error);
    throw error;
  }
};

export const generateCustomVisualization = async (filePath, type, params = {}) => {
  try {
    const response = await api.post('/visualization/visualize', {
      file_path: filePath,
      type: type,
      params: params
    });
    return response.data;
  } catch (error) {
    console.error('Error generating visualization:', error);
    throw error;
  }
};

// Create a default export with all API functions
const apiService = {
  sendMessage,
  stopGeneration,
  getChatHistory,
  listModels,
  switchModel,
  uploadFile,
  listFiles,
  deleteFile,
  getDashboards,
  getDashboard,
  createDashboard,
  updateDashboard,
  deleteDashboard,
  sendMessageWithFiles,
  generateDashboard,
  generateCustomVisualization
};

export default apiService; 