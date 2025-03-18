import api from './apiService';

// Get all dashboards
export const getDashboards = async (category = 'recent') => {
  try {
    const response = await api.get(`/dashboards?category=${category}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboards:', error);
    throw error;
  }
};

// Get a specific dashboard
export const getDashboard = async (dashboardId) => {
  try {
    const response = await api.get(`/dashboards/${dashboardId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard:', error);
    throw error;
  }
};

// Create a new dashboard
export const createDashboard = async (dashboardData) => {
  try {
    const response = await api.post('/dashboards', dashboardData);
    return response.data;
  } catch (error) {
    console.error('Error creating dashboard:', error);
    throw error;
  }
};

// Update a dashboard
export const updateDashboard = async (dashboardId, dashboardData) => {
  try {
    const response = await api.put(`/dashboards/${dashboardId}`, dashboardData);
    return response.data;
  } catch (error) {
    console.error('Error updating dashboard:', error);
    throw error;
  }
};

// Delete a dashboard
export const deleteDashboard = async (dashboardId) => {
  try {
    const response = await api.delete(`/dashboards/${dashboardId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting dashboard:', error);
    throw error;
  }
};

// Share a dashboard
export const shareDashboard = async (dashboardId, shareData) => {
  try {
    const response = await api.post(`/dashboards/${dashboardId}/share`, shareData);
    return response.data;
  } catch (error) {
    console.error('Error sharing dashboard:', error);
    throw error;
  }
};

// Get dashboard sharing settings
export const getDashboardSharing = async (dashboardId) => {
  try {
    const response = await api.get(`/dashboards/${dashboardId}/share`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard sharing settings:', error);
    throw error;
  }
};

// Create a chart for a dashboard
export const createChart = async (dashboardId, chartData) => {
  try {
    const response = await api.post(`/dashboards/${dashboardId}/charts`, chartData);
    return response.data;
  } catch (error) {
    console.error('Error creating chart:', error);
    throw error;
  }
};

// Update a chart
export const updateChart = async (dashboardId, chartId, chartData) => {
  try {
    const response = await api.put(`/dashboards/${dashboardId}/charts/${chartId}`, chartData);
    return response.data;
  } catch (error) {
    console.error('Error updating chart:', error);
    throw error;
  }
};

// Delete a chart
export const deleteChart = async (dashboardId, chartId) => {
  try {
    const response = await api.delete(`/dashboards/${dashboardId}/charts/${chartId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting chart:', error);
    throw error;
  }
};

// Export dashboard as PDF
export const exportDashboardAsPDF = async (dashboardId) => {
  try {
    const response = await api.get(`/dashboards/${dashboardId}/export/pdf`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error('Error exporting dashboard as PDF:', error);
    throw error;
  }
};

// Export dashboard as image
export const exportDashboardAsImage = async (dashboardId, format = 'png') => {
  try {
    const response = await api.get(`/dashboards/${dashboardId}/export/image?format=${format}`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error(`Error exporting dashboard as ${format}:`, error);
    throw error;
  }
};

// Get dashboard templates
export const getDashboardTemplates = async () => {
  try {
    const response = await api.get('/dashboards/templates');
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard templates:', error);
    throw error;
  }
};

// Create dashboard from template
export const createDashboardFromTemplate = async (templateId, customizations) => {
  try {
    const response = await api.post(`/dashboards/templates/${templateId}`, customizations);
    return response.data;
  } catch (error) {
    console.error('Error creating dashboard from template:', error);
    throw error;
  }
};

// Create a named export object
const dashboardService = {
  getDashboards,
  getDashboard,
  createDashboard,
  updateDashboard,
  deleteDashboard,
  shareDashboard,
  getDashboardSharing,
  createChart,
  updateChart,
  deleteChart,
  exportDashboardAsPDF,
  exportDashboardAsImage,
  getDashboardTemplates,
  createDashboardFromTemplate
};

export default dashboardService; 