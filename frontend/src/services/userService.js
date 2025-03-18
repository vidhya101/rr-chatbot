/**
 * User Service
 * 
 * This service handles user authentication and identification.
 */

import api from './apiService';

// Get user ID from local storage
export const getUserId = () => {
  return localStorage.getItem('userId');
};

// Set user ID in local storage
export const setUserId = (userId) => {
  localStorage.setItem('userId', userId);
};

// Get user session ID from local storage
export const getSessionId = () => {
  return localStorage.getItem('sessionId');
};

// Set user session ID in local storage
export const setSessionId = (sessionId) => {
  localStorage.setItem('sessionId', sessionId);
};

// Get user name from local storage
export const getUserName = () => {
  return localStorage.getItem('userName') || 'Guest';
};

// Set user name in local storage
export const setUserName = (userName) => {
  localStorage.setItem('userName', userName);
};

// Get user email from local storage
export const getUserEmail = () => {
  return localStorage.getItem('userEmail');
};

// Set user email in local storage
export const setUserEmail = (userEmail) => {
  localStorage.setItem('userEmail', userEmail);
};

// Get user preferences from local storage
export const getUserPreferences = () => {
  const preferencesString = localStorage.getItem('userPreferences');
  return preferencesString ? JSON.parse(preferencesString) : {
    darkMode: false,
    activeModel: 'mistral',
    colorScheme: 'blue',
  };
};

// Set user preferences in local storage
export const setUserPreferences = (preferences) => {
  localStorage.setItem('userPreferences', JSON.stringify(preferences));
};

// Get user profile
export const getUserProfile = async () => {
  try {
    const response = await api.get('/users/profile');
    return response.data;
  } catch (error) {
    console.error('Error fetching user profile:', error);
    throw error;
  }
};

// Update user profile
export const updateProfile = async (profileData) => {
  try {
    const response = await api.put('/users/profile', profileData);
    return response.data;
  } catch (error) {
    console.error('Error updating profile:', error);
    throw error;
  }
};

// Upload profile picture
export const uploadProfilePicture = async (formData) => {
  try {
    const response = await api.post('/users/profile/picture', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading profile picture:', error);
    throw error;
  }
};

// Update user settings
export const updateUserSettings = async (settings) => {
  try {
    const response = await api.put('/users/settings', settings);
    return response.data;
  } catch (error) {
    console.error('Error updating user settings:', error);
    throw error;
  }
};

// Get user settings
export const getUserSettings = async () => {
  try {
    const response = await api.get('/users/settings');
    return response.data;
  } catch (error) {
    console.error('Error fetching user settings:', error);
    throw error;
  }
};

// Delete user account
export const deleteAccount = async () => {
  try {
    const response = await api.delete('/users/account');
    return response.data;
  } catch (error) {
    console.error('Error deleting account:', error);
    throw error;
  }
};

// Save chat to history
export const saveChat = async (chatData) => {
  try {
    const response = await api.post('/users/chats', chatData);
    return response.data;
  } catch (error) {
    console.error('Error saving chat:', error);
    throw error;
  }
};

// Get chat history
export const getChatHistory = async () => {
  try {
    const response = await api.get('/users/chats');
    return response.data;
  } catch (error) {
    console.error('Error fetching chat history:', error);
    throw error;
  }
};

// Clear chat history
export const clearChatHistory = async () => {
  try {
    const response = await api.delete('/users/chats');
    return response.data;
  } catch (error) {
    console.error('Error clearing chat history:', error);
    throw error;
  }
};

// Get user activity
export const getUserActivity = async () => {
  try {
    const response = await api.get('/users/activity');
    return response.data;
  } catch (error) {
    console.error('Error fetching user activity:', error);
    throw error;
  }
};

// Get user statistics
export const getUserStats = async () => {
  try {
    const response = await api.get('/users/stats');
    return response.data;
  } catch (error) {
    console.error('Error fetching user stats:', error);
    throw error;
  }
};

// Link social account
export const linkSocialAccount = async (provider, code) => {
  try {
    const response = await api.post(`/users/social/${provider}/link`, { code });
    return response.data;
  } catch (error) {
    console.error(`Error linking ${provider} account:`, error);
    throw error;
  }
};

// Unlink social account
export const unlinkSocialAccount = async (provider) => {
  try {
    const response = await api.post(`/users/social/${provider}/unlink`);
    return response.data;
  } catch (error) {
    console.error(`Error unlinking ${provider} account:`, error);
    throw error;
  }
};

// Export user data
export const exportUserData = async () => {
  try {
    const response = await api.get('/users/export', {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error('Error exporting user data:', error);
    throw error;
  }
};

// Log user activity
export const logUserActivity = (activity) => {
  const userId = getUserId();
  const timestamp = new Date().toISOString();
  
  console.log(`User activity: ${userId} - ${timestamp} - ${activity}`);
  
  // In a real application, you would send this to the server
  // For now, we'll just log it to the console
};

// Check if user is logged in
export const isLoggedIn = () => {
  return !!getUserEmail();
};

// Log out user
export const logoutUser = () => {
  // Keep userId for continuity
  const userId = getUserId();
  
  // Clear all other user data
  localStorage.removeItem('sessionId');
  localStorage.removeItem('userName');
  localStorage.removeItem('userEmail');
  
  // Keep preferences and history
  
  return userId;
};

// Create a named export object
const userService = {
  getUserId,
  setUserId,
  getSessionId,
  setSessionId,
  getUserName,
  setUserName,
  getUserEmail,
  setUserEmail,
  getUserPreferences,
  setUserPreferences,
  getUserProfile,
  updateProfile,
  uploadProfilePicture,
  updateUserSettings,
  getUserSettings,
  deleteAccount,
  saveChat,
  getChatHistory,
  clearChatHistory,
  getUserActivity,
  getUserStats,
  linkSocialAccount,
  unlinkSocialAccount,
  exportUserData,
  logUserActivity,
  isLoggedIn,
  logoutUser
};

export default userService; 