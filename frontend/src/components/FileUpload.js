import React, { useState, useRef } from 'react';
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

      for (let i = 0; i < totalFiles; i++) {
        const formData = new FormData();
        formData.append('file', files[i]);

        const response = await fetch('/api/data/upload', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to upload file');
        }

        if (data.success) {
          uploadedFiles.push(data.file);
        }

        // Update progress
        setUploadProgress(((i + 1) / totalFiles) * 100);
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
      setSnackbar({
        open: true,
        message: error.message || 'Failed to upload files',
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
      
      const uploadResponse = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData
      });
      
      const uploadData = await uploadResponse.json();
      
      if (!uploadResponse.ok) {
        throw new Error(uploadData.error || 'Failed to upload file');
      }
      
      if (uploadData.success) {
        const visualizeResponse = await fetch('/api/data/visualize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            fileId: uploadData.file.id,
            type: 'auto'
          })
        });
        
        const visualizeData = await visualizeResponse.json();
        
        if (!visualizeResponse.ok) {
          throw new Error(visualizeData.error || 'Failed to create visualization');
        }
        
        if (visualizeData.success) {
          // Open visualization in new window/tab
          window.open(`/api/data/visualization/${visualizeData.visualization.id}`, '_blank');
        }
      }
    } catch (err) {
      console.error('Error visualizing file:', err);
      setSnackbar({
        open: true,
        message: err.message || 'Error preparing file for visualization',
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