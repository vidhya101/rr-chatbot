/**
 * Application configuration
 */

// API configuration
export const API_BASE_URL = 'http://localhost:5000/api';

// Debug settings
export const DEBUG = true; // Set to false in production

// Utility functions
export const debugLog = (message, data) => {
  if (DEBUG) {
    console.log(`[DEBUG] ${message}`, data);
  }
};

// API utility functions
export const fetchWithErrorHandling = async (url, options = {}) => {
  try {
    debugLog(`Fetching: ${url}`, options);
    const response = await fetch(url, options);
    const data = await response.json();
    debugLog(`Response from ${url}:`, data);
    
    if (!response.ok) {
      throw new Error(data.error || `Request failed with status ${response.status}`);
    }
    
    return data;
  } catch (error) {
    debugLog(`Error fetching ${url}:`, error);
    throw error;
  }
};

// Export other configuration as needed
export const APP_CONFIG = {
  appName: 'RR-Chatbot',
  version: '1.0.0',
  apiUrl: API_BASE_URL
};

export default APP_CONFIG; 