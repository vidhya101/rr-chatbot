import React, { useState, useRef } from 'react';
import axios from 'axios';
import './FileUpload.css';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  CircularProgress, 
  IconButton, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import BarChartIcon from '@mui/icons-material/BarChart';

const UploadBox = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  textAlign: 'center',
  cursor: 'pointer',
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' ? '#2a2a2a' : '#e0e0e0',
  },
  transition: 'background-color 0.3s',
}));

const HiddenInput = styled('input')({
  display: 'none',
});

const FileUpload = ({ onFileUpload, onFileSelect }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewFile, setPreviewFile] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showVisualization, setShowVisualization] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  };

  const handleFileInputChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    handleFiles(selectedFiles);
  };

  const handleFiles = (newFiles) => {
    // Add new files to the list
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
    
    // If onFileSelect callback is provided, call it with the first file
    if (onFileSelect && newFiles.length > 0) {
      onFileSelect(newFiles[0]);
    }
    
    setSnackbar({
      open: true,
      message: `${newFiles.length} file(s) added successfully`,
      severity: 'success'
    });
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setSnackbar({
        open: true,
        message: 'Please select files to upload',
        severity: 'warning'
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const totalFiles = files.length;
      const uploadedFiles = [];
      
      console.log('Starting file upload...');

      for (let i = 0; i < totalFiles; i++) {
        const formData = new FormData();
        formData.append('file', files[i]);
        
        console.log(`Uploading file ${i+1}/${totalFiles}: ${files[i].name}`);

        // Use direct axios call with full URL
        const response = await axios.post('http://localhost:5000/api/data/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            console.log(`Upload progress: ${percentCompleted}%`);
            setUploadProgress(percentCompleted);
          }
        });
        
        console.log('Upload response:', response.data);

        if (response.data.success && response.data.file) {
          uploadedFiles.push(response.data.file);
        } else {
          throw new Error(response.data.error || 'Failed to upload file');
        }
      }

      // Clear files list after successful upload
      setFiles([]);
      setUploadProgress(0);

      // Call onFileUpload callback with uploaded files
      if (onFileUpload) {
        onFileUpload(uploadedFiles);
      }

      setSnackbar({
        open: true,
        message: 'All files uploaded successfully',
        severity: 'success'
      });
    } catch (error) {
      console.error('Upload error:', error);
      
      // Provide more detailed error messages
      let errorMessage = 'Failed to upload files';
      if (error.response && error.response.data) {
        errorMessage = error.response.data.error || error.response.data.message || errorMessage;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setSnackbar({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setUploading(false);
    }
  };

  const handleRemoveFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const handlePreviewFile = (file) => {
    setPreviewFile(file);
    setShowPreview(true);
  };

  const handleClosePreview = () => {
    setShowPreview(false);
    setPreviewFile(null);
  };

  const handleVisualizeFile = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      console.log(`Visualizing file: ${file.name}`);
      
      // Use direct axios call with full URL for upload
      console.log('Uploading file for visualization...');
      const uploadResponse = await axios.post('http://localhost:5000/api/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log('Upload response:', uploadResponse.data);
      
      if (!uploadResponse.data.success) {
        throw new Error(uploadResponse.data.error || 'Failed to upload file');
      }
      
      if (uploadResponse.data.file) {
        console.log('Creating visualization...');
        const visualizeResponse = await axios.post('http://localhost:5000/api/data/visualize', {
          fileId: uploadResponse.data.file.id,
          type: 'auto'
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        console.log('Visualization response:', visualizeResponse.data);
        
        if (!visualizeResponse.data.success) {
          throw new Error(visualizeResponse.data.error || 'Failed to create visualization');
        }
        
        if (visualizeResponse.data.visualization) {
          const visualizationUrl = `http://localhost:5000/api/data/visualization/${visualizeResponse.data.visualization.id}`;
          console.log('Opening visualization URL:', visualizationUrl);
          // Open visualization in new window/tab
          window.open(visualizationUrl, '_blank');
        }
      }
    } catch (err) {
      console.error('Error visualizing file:', err);
      
      // Provide more detailed error messages
      let errorMessage = 'Error preparing file for visualization';
      if (err.response && err.response.data) {
        errorMessage = err.response.data.error || err.response.data.message || errorMessage;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setSnackbar({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  return (
    <Box>
      <Box
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        sx={{ mb: 3 }}
      >
        <UploadBox>
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileInputChange}
            multiple
          />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Drag & Drop Files Here
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            or
          </Typography>
          <Button
            variant="contained"
            onClick={() => fileInputRef.current?.click()}
          >
            Browse Files
          </Button>
          <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
            Supported formats: CSV, Excel, JSON, Parquet
          </Typography>
        </UploadBox>
      </Box>

      {files.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Selected Files
          </Typography>
          <List>
            {files.map((file, index) => (
              <ListItem key={index} divider>
                <ListItemText
                  primary={file.name}
                  secondary={`Size: ${(file.size / 1024).toFixed(2)} KB`}
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    aria-label="visualize"
                    onClick={() => handleVisualizeFile(file)}
                    sx={{ mr: 1 }}
                  >
                    <BarChartIcon />
                  </IconButton>
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleRemoveFile(index)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          <Box mt={2} display="flex" justifyContent="flex-end">
            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={uploading}
              startIcon={uploading ? <CircularProgress size={20} /> : null}
            >
              {uploading ? `Uploading ${uploadProgress.toFixed(0)}%` : 'Upload Files'}
            </Button>
          </Box>
        </Box>
      )}

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default FileUpload; 