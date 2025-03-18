import React from 'react';
import { Link } from 'react-router-dom';
import './NotFound.css';

// Material UI components
import {
  Container,
  Typography,
  Button,
  Paper
} from '@mui/material';

// Material UI Icons
import HomeIcon from '@mui/icons-material/Home';
import SentimentDissatisfiedIcon from '@mui/icons-material/SentimentDissatisfied';

const NotFound = ({ darkMode }) => {
  return (
    <div className={`not-found-container ${darkMode ? 'dark-mode' : ''}`}>
      <Container maxWidth="md">
        <Paper elevation={3} className="not-found-paper">
          <div className="not-found-content">
            <SentimentDissatisfiedIcon className="not-found-icon" />
            
            <Typography variant="h1" className="not-found-code">
              404
            </Typography>
            
            <Typography variant="h4" className="not-found-title">
              Page Not Found
            </Typography>
            
            <Typography variant="body1" className="not-found-message">
              The page you are looking for doesn't exist or has been moved.
            </Typography>
            
            <div className="not-found-actions">
              <Button
                variant="contained"
                color="primary"
                component={Link}
                to="/"
                startIcon={<HomeIcon />}
                className="home-button"
              >
                Back to Home
              </Button>
            </div>
          </div>
        </Paper>
      </Container>
    </div>
  );
};

export default NotFound; 