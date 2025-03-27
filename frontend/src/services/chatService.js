import api from '../utils/api';

/**
 * Send a message to the AI and get a response
 * @param {string} message - The message to send
 * @param {string} chatId - Optional chat ID for continuing a conversation
 * @param {string} model - Optional model ID to use for the response
 * @returns {Promise<Object>} - The response from the AI
 */
export const sendMessage = async (message, chatId = null, model = null) => {
  try {
    const payload = { 
      message,
      ...(chatId && { chatId }),
      ...(model && { model })
    };
    
    // Try the public endpoint first
    try {
      const response = await api.post('/public/chat', payload);
      
      if (!response.data.success) {
        throw new Error(response.data.error || 'Failed to get response from AI');
      }
      
      return {
        message: response.data.response,
        model: response.data.model,
        processingTime: response.data.processing_time
      };
    } catch (publicError) {
      // If public endpoint fails, try the authenticated endpoint
      if (chatId) {
        const authResponse = await api.post(`/api/chats/${chatId}/messages`, payload);
        return {
          message: authResponse.data.message,
          model: authResponse.data.model,
          processingTime: authResponse.data.processing_time
        };
      }
      throw publicError;
    }
  } catch (error) {
    console.error('Error sending message:', error);
    throw new Error(
      error.response?.data?.error || 
      error.message || 
      'Failed to send message. Please try again.'
    );
  }
};

/**
 * Get all chats for the current user
 * @returns {Promise<Array>} - Array of chat objects
 */
export const getChats = async () => {
  try {
    const response = await api.get('/api/chats');
    return response.data.chats;
  } catch (error) {
    console.error('Error getting chats:', error);
    throw error;
  }
};

/**
 * Get a specific chat with all messages
 * @param {string} chatId - The ID of the chat to get
 * @returns {Promise<Object>} - The chat object with messages
 */
export const getChat = async (chatId) => {
  try {
    const response = await api.get(`/api/chats/${chatId}`);
    return response.data.chat;
  } catch (error) {
    console.error('Error getting chat:', error);
    throw error;
  }
};

/**
 * Delete a chat
 * @param {string} chatId - The ID of the chat to delete
 * @returns {Promise<Object>} - The response from the server
 */
export const deleteChat = async (chatId) => {
  try {
    const response = await api.delete(`/api/chats/${chatId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting chat:', error);
    throw error;
  }
};

/**
 * Update the title of a chat
 * @param {string} chatId - The ID of the chat to update
 * @param {string} title - The new title for the chat
 * @returns {Promise<Object>} - The response from the server
 */
export const updateChatTitle = async (chatId, title) => {
  try {
    const response = await api.put(`/api/chats/${chatId}/title`, { title });
    return response.data;
  } catch (error) {
    console.error('Error updating chat title:', error);
    throw error;
  }
};

/**
 * Get available AI models
 * @returns {Promise<Array>} - Array of model objects
 */
export const getModels = async () => {
  try {
    const response = await api.get('/api/models');
    return response.data.models;
  } catch (error) {
    console.error('Error getting models:', error);
    throw error;
  }
};

export default {
  sendMessage,
  getChats,
  getChat,
  deleteChat,
  updateChatTitle,
  getModels
}; 