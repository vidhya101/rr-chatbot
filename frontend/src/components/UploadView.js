import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { API_BASE_URL } from '../utils/config';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Chip
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  GetApp as DownloadIcon,
  Share as ShareIcon,
  Description as FileIcon,
  PictureAsPdf as PdfIcon,
  TableChart as CsvIcon,
  Code as JsonIcon
} from '@mui/icons-material';

const UploadView = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [successAlert, setSuccessAlert] = useState(false);
  const fileInputRef = useRef();

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/data/files`);
      if (!response.ok) throw new Error('Failed to fetch files');

      const data = await response.json();
      setFiles(data.files || []);
    } catch (err) {
      console.error('Error fetching files:', err);
      setError('Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleUpload(file);
    }
  };

  const handleUpload = async (file) => {
    setUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Starting file upload with axios...');
      
      // Use the centralized API URL
      const response = await axios.post(`${API_BASE_URL}/data/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log(`Upload progress: ${progress}%`);
          setUploadProgress(progress);
        }
      });
      
      console.log('Upload response:', response.data);
      
      if (response.data.success) {
        setFiles([...files, response.data.file]);
        
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        
        // Show success message with option to go to Data Modeling
        setSuccessAlert(true);
      } else {
        throw new Error(response.data.error || 'Failed to upload file');
      }
    } catch (err) {
      console.error('Error uploading file:', err);
      
      // Provide more detailed error messages
      let errorMessage = 'Failed to upload file';
      if (err.response && err.response.data) {
        errorMessage = err.response.data.error || err.response.data.message || errorMessage;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (fileId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/data/files/${fileId}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete file');

      setFiles(files.filter(file => file.id !== fileId));
    } catch (err) {
      console.error('Error deleting file:', err);
      setError('Failed to delete file');
    }
  };

  const handleDownload = async (file) => {
    try {
      const response = await fetch(`${API_BASE_URL}/data/files/${file.id}/download`);
      if (!response.ok) throw new Error('Failed to download file');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading file:', err);
      setError('Failed to download file');
    }
  };

  const getFileIcon = (fileType) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return <PdfIcon />;
      case 'csv':
        return <CsvIcon />;
      case 'json':
        return <JsonIcon />;
      default:
        return <FileIcon />;
    }
  };

  const renderPreviewDialog = () => (
    <Dialog
      open={previewDialogOpen}
      onClose={() => setPreviewDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        {selectedFile?.name}
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="subtitle1">File Details</Typography>
            <Typography variant="body2">
              Type: {selectedFile?.type}
            </Typography>
            <Typography variant="body2">
              Size: {(selectedFile?.size / 1024).toFixed(2)} KB
            </Typography>
            <Typography variant="body2">
              Uploaded: {new Date(selectedFile?.uploaded_at).toLocaleString()}
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Typography variant="subtitle1">Preview</Typography>
            {/* Add file preview based on file type */}
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setPreviewDialogOpen(false)}>Close</Button>
        <Button
          startIcon={<DownloadIcon />}
          onClick={() => handleDownload(selectedFile)}
        >
          Download
        </Button>
      </DialogActions>
    </Dialog>
  );

  const handleCloseSuccessAlert = () => {
    setSuccessAlert(false);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Upload Data
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            p={3}
            sx={{
              border: '2px dashed #ccc',
              borderRadius: 1,
              backgroundColor: '#fafafa',
              cursor: 'pointer'
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />
            <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Click to Upload Files
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Support for CSV, PDF, JSON, and other data files
            </Typography>
          </Box>

          {uploading && (
            <Box sx={{ width: '100%', mt: 2 }}>
              <LinearProgress variant="determinate" value={uploadProgress} />
              <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 1 }}>
                Uploading... {uploadProgress}%
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {successAlert && (
        <Alert 
          severity="success" 
          sx={{ mb: 2 }}
          action={
            <Box display="flex" alignItems="center">
              <Button 
                color="inherit" 
                size="small" 
                onClick={handleCloseSuccessAlert}
                sx={{ mr: 1 }}
              >
                Close
              </Button>
              <Button 
                variant="outlined" 
                color="inherit" 
                size="small" 
                component={Link}
                to="/modeling"
              >
                Go to Data Modeling
              </Button>
            </Box>
          }
        >
          File uploaded successfully! You can now process it in the Data Modeling section.
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Uploaded Files
          </Typography>
          <List>
            {files.map((file, index) => (
              <React.Fragment key={file.id}>
                <ListItem>
                  <ListItemIcon>
                    {getFileIcon(file.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={file.name}
                    secondary={
                      <>
                        <Typography variant="body2" color="textSecondary">
                          Size: {(file.size / 1024).toFixed(2)} KB
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Uploaded: {new Date(file.uploaded_at).toLocaleString()}
                        </Typography>
                      </>
                    }
                  />
                  <Box>
                    <IconButton
                      onClick={() => {
                        setSelectedFile(file);
                        setPreviewDialogOpen(true);
                      }}
                    >
                      <ViewIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDownload(file)}>
                      <DownloadIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(file.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </ListItem>
                {index < files.length - 1 && <Divider />}
              </React.Fragment>
            ))}
            {files.length === 0 && (
              <ListItem>
                <ListItemText
                  primary="No files uploaded"
                  secondary="Upload files to get started"
                />
              </ListItem>
            )}
          </List>
        </CardContent>
      </Card>

      {renderPreviewDialog()}
    </Box>
  );
};

export default UploadView; 