import React, { useState, useRef, useEffect } from 'react';
import { Box, Button, Typography, Paper, List, ListItem, ListItemText, 
         ListItemIcon, ListItemSecondaryAction, IconButton, Chip, 
         Dialog, DialogTitle, DialogContent, DialogActions, LinearProgress,
         Alert, Snackbar, Tooltip, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import TableChartIcon from '@mui/icons-material/TableChart';
import CodeIcon from '@mui/icons-material/Code';
import TextSnippetIcon from '@mui/icons-material/TextSnippet';
import PreviewIcon from '@mui/icons-material/Preview';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import UploadProgress from './UploadProgress';

// ... existing styled components ...

const FileUploader = ({ onUpload, maxFileSize = 50 * 1024 * 1024, onValidationError }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [validationErrors, setValidationErrors] = useState({});
  const [previewDialog, setPreviewDialog] = useState({ open: false, content: null });
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  // ... existing getFileIcon and getFileSize functions ...

  // Validate file
  const validateFile = (file) => {
    const errors = [];
    
    // Check file size
    if (file.size > maxFileSize) {
      errors.push(`File size exceeds ${maxFileSize / (1024 * 1024)}MB limit`);
    }
    
    // Check file type
    const fileType = file.name.split('.').pop().toLowerCase();
    const allowedTypes = ['csv', 'xlsx', 'xls', 'json', 'txt'];
    if (!allowedTypes.includes(fileType)) {
      errors.push('Unsupported file type');
    }
    
    // Check file name
    if (!/^[a-zA-Z0-9._-]+$/.test(file.name)) {
      errors.push('Invalid file name characters');
    }
    
    return errors;
  };

  // Handle file selection with validation
  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    const newFiles = [];
    const newErrors = { ...validationErrors };
    
    files.forEach(file => {
      const errors = validateFile(file);
      if (errors.length === 0) {
        newFiles.push(file);
      } else {
        newErrors[file.name] = errors;
      }
    });
    
    setSelectedFiles([...selectedFiles, ...newFiles]);
    setValidationErrors(newErrors);
    
    if (Object.keys(newErrors).length > 0) {
      onValidationError?.(newErrors);
    }
  };

  // Handle drag and drop with validation
  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    const newFiles = [];
    const newErrors = { ...validationErrors };
    
    files.forEach(file => {
      const errors = validateFile(file);
      if (errors.length === 0) {
        newFiles.push(file);
      } else {
        newErrors[file.name] = errors;
      }
    });
    
    setSelectedFiles([...selectedFiles, ...newFiles]);
    setValidationErrors(newErrors);
    
    if (Object.keys(newErrors).length > 0) {
      onValidationError?.(newErrors);
    }
  };

  // Preview file content
  const handlePreview = async (file) => {
    try {
      const content = await readFileContent(file);
      setPreviewDialog({ open: true, content });
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Error previewing file',
        severity: 'error'
      });
    }
  };

  // Read file content for preview
  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target.result;
        resolve(content);
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  // Handle upload with progress tracking
  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    
    setIsUploading(true);
    const progress = {};
    
    try {
      for (const file of selectedFiles) {
        progress[file.name] = 0;
        setUploadProgress(progress);
        
        // Simulate upload progress
        for (let i = 0; i <= 100; i += 10) {
          await new Promise(resolve => setTimeout(resolve, 200));
          progress[file.name] = i;
          setUploadProgress({ ...progress });
        }
      }
      
      onUpload(selectedFiles);
      setSnackbar({
        open: true,
        message: 'Files uploaded successfully',
        severity: 'success'
      });
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Error uploading files',
        severity: 'error'
      });
    } finally {
      setIsUploading(false);
      setUploadProgress({});
    }
  };

  // Get file status icon
  const getFileStatusIcon = (file) => {
    if (validationErrors[file.name]) {
      return <ErrorIcon color="error" />;
    }
    if (uploadProgress[file.name] === 100) {
      return <CheckCircleIcon color="success" />;
    }
    if (uploadProgress[file.name] > 0) {
      return <CircularProgress size={20} value={uploadProgress[file.name]} />;
    }
    return null;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Upload Data Files
      </Typography>
      
      <UploadBox
        onClick={handleBoxClick}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        sx={{ opacity: isUploading ? 0.5 : 1 }}
      >
        <CloudUploadIcon fontSize="large" color="primary" />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Drag & Drop Files Here
        </Typography>
        <Typography variant="body2" color="textSecondary">
          or click to browse
        </Typography>
        <VisuallyHiddenInput
          type="file"
          multiple
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv,.xlsx,.xls,.json,.txt"
          disabled={isUploading}
        />
      </UploadBox>
      
      {selectedFiles.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Selected Files ({selectedFiles.length})
          </Typography>
          
          <List>
            {selectedFiles.map((file, index) => (
              <ListItem 
                key={index} 
                sx={{ 
                  bgcolor: 'background.paper', 
                  mb: 1,
                  border: validationErrors[file.name] ? '1px solid error.main' : 'none'
                }}
              >
                <ListItemIcon>
                  {getFileIcon(file)}
                  {getFileStatusIcon(file)}
                </ListItemIcon>
                <ListItemText 
                  primary={file.name} 
                  secondary={
                    <Box>
                      <Typography variant="body2">
                        {getFileSize(file.size)}
                      </Typography>
                      {validationErrors[file.name] && (
                        <Typography variant="caption" color="error">
                          {validationErrors[file.name].join(', ')}
                        </Typography>
                      )}
                      {uploadProgress[file.name] > 0 && (
                        <LinearProgress 
                          variant="determinate" 
                          value={uploadProgress[file.name]} 
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Chip 
                    label={file.name.split('.').pop().toUpperCase()} 
                    size="small" 
                    color="primary" 
                    variant="outlined" 
                    sx={{ mr: 1 }}
                  />
                  <Tooltip title="Preview">
                    <IconButton 
                      edge="end" 
                      aria-label="preview" 
                      onClick={() => handlePreview(file)}
                      sx={{ mr: 1 }}
                    >
                      <PreviewIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete">
                    <IconButton 
                      edge="end" 
                      aria-label="delete" 
                      onClick={() => handleDeleteFile(index)}
                      disabled={isUploading}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          
          <Button
            variant="contained"
            startIcon={<AttachFileIcon />}
            onClick={handleUpload}
            fullWidth
            sx={{ mt: 2 }}
            disabled={isUploading || Object.keys(validationErrors).length > 0}
          >
            {isUploading ? 'Uploading...' : 'Upload Files'}
          </Button>
        </Box>
      )}
      
      <Box sx={{ mt: 3 }}>
        <Typography variant="body2" color="textSecondary">
          Supported file formats: CSV, Excel (XLSX, XLS), JSON, TXT
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Maximum file size: {maxFileSize / (1024 * 1024)}MB
        </Typography>
      </Box>

      {/* Preview Dialog */}
      <Dialog
        open={previewDialog.open}
        onClose={() => setPreviewDialog({ open: false, content: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>File Preview</DialogTitle>
        <DialogContent>
          <pre style={{ 
            whiteSpace: 'pre-wrap', 
            wordWrap: 'break-word',
            maxHeight: '60vh',
            overflow: 'auto'
          }}>
            {previewDialog.content}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialog({ open: false, content: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default FileUploader; 