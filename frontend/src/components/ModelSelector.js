import React, { useState, useEffect } from 'react';
import { Select, MenuItem, FormControl, InputLabel, Box, Typography, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';
import chatService from '../services/chatService';

// Styled components
const ModelChip = styled(Chip)(({ theme, provider }) => ({
  margin: theme.spacing(0.5),
  backgroundColor: 
    provider === 'openai' ? '#10a37f' : 
    provider === 'ollama' ? '#ff6b6b' :
    provider === 'mistral' ? '#7e57c2' :
    provider === 'huggingface' ? '#ffb74d' : 
    theme.palette.primary.main,
  color: '#fff',
  '& .MuiChip-label': {
    paddingLeft: 6,
    paddingRight: 6,
  }
}));

const ModelSelector = ({ selectedModel, onModelChange }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const modelData = await chatService.getModels();
        setModels(modelData);
        
        // If no model is selected yet, select the default one
        if (!selectedModel && modelData.length > 0) {
          const defaultModel = modelData.find(model => model.isDefault) || modelData[0];
          onModelChange(defaultModel.id);
        }
      } catch (err) {
        console.error('Error fetching models:', err);
        setError('Failed to load models. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [selectedModel, onModelChange]);

  // Group models by provider
  const groupedModels = models.reduce((acc, model) => {
    const provider = model.provider || 'other';
    if (!acc[provider]) {
      acc[provider] = [];
    }
    acc[provider].push(model);
    return acc;
  }, {});

  // Provider labels
  const providerLabels = {
    'openai': 'OpenAI',
    'ollama': 'Ollama (Local)',
    'mistral': 'Mistral AI',
    'huggingface': 'Hugging Face',
    'other': 'Other'
  };

  if (loading) {
    return <Typography variant="body2">Loading models...</Typography>;
  }

  if (error) {
    return <Typography color="error" variant="body2">{error}</Typography>;
  }

  return (
    <Box sx={{ minWidth: 200, mb: 2 }}>
      <FormControl fullWidth size="small">
        <InputLabel id="model-select-label">AI Model</InputLabel>
        <Select
          labelId="model-select-label"
          id="model-select"
          value={selectedModel || ''}
          label="AI Model"
          onChange={(e) => onModelChange(e.target.value)}
          renderValue={(selected) => {
            const model = models.find(m => m.id === selected);
            if (!model) return selected;
            return (
              <ModelChip 
                label={model.name} 
                provider={model.provider}
                size="small"
              />
            );
          }}
        >
          {Object.entries(groupedModels).map(([provider, providerModels]) => [
            <MenuItem key={provider} disabled divider>
              <Typography variant="caption" color="textSecondary">
                {providerLabels[provider] || provider}
              </Typography>
            </MenuItem>,
            ...providerModels.map(model => (
              <MenuItem key={model.id} value={model.id}>
                <ModelChip 
                  label={model.name} 
                  provider={model.provider}
                  size="small"
                />
              </MenuItem>
            ))
          ]).flat()}
        </Select>
      </FormControl>
    </Box>
  );
};

export default ModelSelector; 