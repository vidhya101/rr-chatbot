/**
 * Create an interactive visualization with advanced features
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationType - Type of visualization to create
 * @param {Object} params - Additional parameters for the visualization
 * @returns {Promise<Object>} Response with interactive visualization details
 */
async createInteractiveVisualization(datasetId, visualizationType, params = {}) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/interactive-visualize`, {
      visualization_type: visualizationType,
      params
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Perform advanced statistical analysis on a dataset
 * @param {string} datasetId - ID of the dataset
 * @param {Object} analysisConfig - Configuration for the analysis
 * @returns {Promise<Object>} Response with advanced analysis results
 */
async performAdvancedAnalysis(datasetId, analysisConfig) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/advanced-analyze`, {
      analysis_config: analysisConfig
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Generate a comprehensive data report
 * @param {string} datasetId - ID of the dataset
 * @param {Object} reportConfig - Configuration for the report
 * @returns {Promise<Object>} Response with report details
 */
async generateDataReport(datasetId, reportConfig = {}) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/report`, {
      report_config: reportConfig
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Create a dashboard with multiple visualizations
 * @param {string} datasetId - ID of the dataset
 * @param {Array} visualizations - Array of visualization configurations
 * @returns {Promise<Object>} Response with dashboard details
 */
async createDashboard(datasetId, visualizations) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/dashboard`, {
      visualizations
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Export visualizations in various formats
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @param {string} format - Export format ('png', 'svg', 'pdf', 'html')
 * @returns {Promise<Object>} Response with export details
 */
async exportVisualization(datasetId, visualizationId, format = 'png') {
  try {
    const response = await this.apiClient.get(
      `/datasets/${datasetId}/visualizations/${visualizationId}/export?format=${format}`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Save a visualization configuration for later use
 * @param {string} datasetId - ID of the dataset
 * @param {Object} visualizationConfig - Configuration to save
 * @returns {Promise<Object>} Response with saved configuration details
 */
async saveVisualizationConfig(datasetId, visualizationConfig) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/visualization-configs`, {
      config: visualizationConfig
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Apply a saved visualization configuration
 * @param {string} datasetId - ID of the dataset
 * @param {string} configId - ID of the saved configuration
 * @returns {Promise<Object>} Response with applied configuration details
 */
async applyVisualizationConfig(datasetId, configId) {
  try {
    const response = await this.apiClient.post(
      `/datasets/${datasetId}/visualization-configs/${configId}/apply`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Get visualization templates
 * @returns {Promise<Object>} Response with available templates
 */
async getVisualizationTemplates() {
  try {
    const response = await this.apiClient.get('/visualization-templates');
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Apply a visualization template
 * @param {string} datasetId - ID of the dataset
 * @param {string} templateId - ID of the template to apply
 * @param {Object} params - Additional parameters for template customization
 * @returns {Promise<Object>} Response with applied template details
 */
async applyVisualizationTemplate(datasetId, templateId, params = {}) {
  try {
    const response = await this.apiClient.post(`/datasets/${datasetId}/templates/${templateId}/apply`, {
      params
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Share a visualization
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @param {Object} shareConfig - Sharing configuration
 * @returns {Promise<Object>} Response with sharing details
 */
async shareVisualization(datasetId, visualizationId, shareConfig) {
  try {
    const response = await this.apiClient.post(
      `/datasets/${datasetId}/visualizations/${visualizationId}/share`,
      { share_config: shareConfig }
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Get visualization sharing status
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @returns {Promise<Object>} Response with sharing status
 */
async getVisualizationSharingStatus(datasetId, visualizationId) {
  try {
    const response = await this.apiClient.get(
      `/datasets/${datasetId}/visualizations/${visualizationId}/sharing-status`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Update visualization sharing settings
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @param {Object} sharingSettings - New sharing settings
 * @returns {Promise<Object>} Response with updated sharing settings
 */
async updateVisualizationSharing(datasetId, visualizationId, sharingSettings) {
  try {
    const response = await this.apiClient.put(
      `/datasets/${datasetId}/visualizations/${visualizationId}/sharing`,
      { sharing_settings: sharingSettings }
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Get visualization analytics
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @returns {Promise<Object>} Response with visualization analytics
 */
async getVisualizationAnalytics(datasetId, visualizationId) {
  try {
    const response = await this.apiClient.get(
      `/datasets/${datasetId}/visualizations/${visualizationId}/analytics`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Schedule visualization updates
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @param {Object} scheduleConfig - Schedule configuration
 * @returns {Promise<Object>} Response with schedule details
 */
async scheduleVisualizationUpdate(datasetId, visualizationId, scheduleConfig) {
  try {
    const response = await this.apiClient.post(
      `/datasets/${datasetId}/visualizations/${visualizationId}/schedule`,
      { schedule_config: scheduleConfig }
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Get scheduled visualization updates
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @returns {Promise<Object>} Response with scheduled updates
 */
async getScheduledUpdates(datasetId, visualizationId) {
  try {
    const response = await this.apiClient.get(
      `/datasets/${datasetId}/visualizations/${visualizationId}/schedule`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
}

/**
 * Cancel scheduled visualization updates
 * @param {string} datasetId - ID of the dataset
 * @param {string} visualizationId - ID of the visualization
 * @param {string} scheduleId - ID of the schedule to cancel
 * @returns {Promise<Object>} Response with cancellation status
 */
async cancelScheduledUpdate(datasetId, visualizationId, scheduleId) {
  try {
    const response = await this.apiClient.delete(
      `/datasets/${datasetId}/visualizations/${visualizationId}/schedule/${scheduleId}`
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    return { success: false, error: error.message };
  }
} 