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
import { CloudUpload as CloudUploadIcon, Delete as DeleteIcon, Visibility as VisibilityIcon, BarChart as BarChartIcon } from '@mui/icons-material';
import { uploadFile } from '../services/apiService';
import Visualization from './Visualization';

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
  const [error, setError] = useState(null);
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

  const handleUploadClick = () => {
    fileInputRef.current.click();
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
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const onUploadProgress = (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      };

      const response = await uploadFile(formData, onUploadProgress);

      if (response.data && response.data.success) {
        // Call the onFileUpload callback if provided
        if (onFileUpload) {
          onFileUpload(response.data.files);
        }
        
        setSnackbar({
          open: true,
          message: 'Files uploaded successfully',
          severity: 'success'
        });
        
        // Clear the file list after successful upload
        setFiles([]);
      } else {
        setError(response.data?.error || 'Upload failed');
        setSnackbar({
          open: true,
          message: response.data?.error || 'Upload failed',
          severity: 'error'
        });
      }
    } catch (err) {
      console.error('Error uploading files:', err);
      setError('Error uploading files. Please try again.');
      setSnackbar({
        open: true,
        message: 'Error uploading files. Please try again.',
        severity: 'error'
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
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
  };

  const handleVisualizeFile = async (file) => {
    try {
      setShowVisualization(true);
      
      // Use the named export uploadFile instead of apiService.uploadFile
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await uploadFile(formData);
      
      if (response.data && response.data.success) {
        setSelectedFile(response.data.files[0]);
      } else {
        setSnackbar({
          open: true,
          message: response.data?.error || 'Failed to upload file for visualization',
          severity: 'error'
        });
      }
    } catch (err) {
      console.error('Error preparing file for visualization:', err);
      setSnackbar({
        open: true,
        message: 'Error preparing file for visualization',
        severity: 'error'
      });
    } finally {
      setShowVisualization(false);
    }
  };

  const handleCloseVisualization = () => {
    setShowVisualization(false);
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const getFilePreview = (file) => {
    if (file.type.startsWith('image/')) {
      return URL.createObjectURL(file);
    }
    return null;
  };

  const isDataFile = (file) => {
    const dataFileExtensions = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.tsv'];
    const fileName = file.name.toLowerCase();
    return dataFileExtensions.some(ext => fileName.endsWith(ext));
  };

  return (
    <Box>
      <UploadBox
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleUploadClick}
      >
        <CloudUploadIcon fontSize="large" color="primary" />
        <Typography variant="h6" gutterBottom>
          Drag & Drop Files Here
        </Typography>
        <Typography variant="body2" color="textSecondary">
          or click to browse
        </Typography>
        <HiddenInput
          type="file"
          multiple
          ref={fileInputRef}
          onChange={handleFileInputChange}
        />
      </UploadBox>

      {files.length > 0 && (
        <Box mt={3}>
          <Typography variant="h6" gutterBottom>
            Selected Files ({files.length})
          </Typography>
          <List>
            {files.map((file, index) => (
              <ListItem key={index} divider>
                <ListItemText
                  primary={file.name}
                  secondary={`${(file.size / 1024).toFixed(2)} KB`}
                />
                <ListItemSecondaryAction>
                  {file.type.startsWith('image/') && (
                    <IconButton 
                      edge="end" 
                      aria-label="preview" 
                      onClick={() => handlePreviewFile(file)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                  )}
                  {isDataFile(file) && (
                    <IconButton 
                      edge="end" 
                      aria-label="visualize" 
                      onClick={() => handleVisualizeFile(file)}
                      sx={{ mr: 1 }}
                    >
                      <BarChartIcon />
                    </IconButton>
                  )}
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
              {uploading ? `Uploading ${uploadProgress}%` : 'Upload Files'}
            </Button>
          </Box>
        </Box>
      )}

      {/* File Preview Dialog */}
      <Dialog open={showPreview} onClose={handleClosePreview} maxWidth="md" fullWidth>
        <DialogTitle>File Preview</DialogTitle>
        <DialogContent>
          {previewFile && previewFile.type.startsWith('image/') && (
            <Box display="flex" justifyContent="center">
              <img 
                src={getFilePreview(previewFile)} 
                alt={previewFile.name} 
                style={{ maxWidth: '100%', maxHeight: '70vh' }} 
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePreview} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Visualization Dialog */}
      <Dialog 
        open={showVisualization} 
        onClose={handleCloseVisualization} 
        maxWidth="lg" 
        fullWidth
      >
        <DialogTitle>Data Visualization</DialogTitle>
        <DialogContent>
          {selectedFile && (
            <Visualization 
              fileData={selectedFile} 
              onClose={handleCloseVisualization} 
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseVisualization} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
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