import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Sidebar.css';

// Material UI Icons
import ChatIcon from '@mui/icons-material/Chat';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderIcon from '@mui/icons-material/Folder';
import BarChartIcon from '@mui/icons-material/BarChart';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

// Services
import { listModels } from '../services/apiService';
import { clearChatHistory } from '../services/userService';

const Sidebar = ({ isOpen, chatHistory, activeModel, switchModel, darkMode }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showModels, setShowModels] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const navigate = useNavigate();

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const modelsList = await listModels();
        setModels(Array.isArray(modelsList) ? modelsList : []);
        setError(null);
      } catch (err) {
        console.error('Error fetching models:', err);
        setError('Failed to load models');
        setModels([]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  // Handle model switch
  const handleModelSwitch = async (model) => {
    try {
      await switchModel(model);
    } catch (err) {
      console.error('Error switching model:', err);
    }
  };

  // Handle clear history
  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear your chat history?')) {
      clearChatHistory();
      navigate('/');
      window.location.reload();
    }
  };

  // Toggle models dropdown
  const toggleModels = () => {
    setShowModels(!showModels);
  };

  // Toggle history dropdown
  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'} ${darkMode ? 'dark-mode' : ''}`}>
      <div className="sidebar-section">
        <Link to="/" className="new-chat-button">
          <ChatIcon />
          <span>New Chat</span>
        </Link>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-header" onClick={toggleModels}>
          <div className="sidebar-section-title">
            <SmartToyIcon />
            <span>Models</span>
          </div>
          {showModels ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
        </div>
        
        {showModels && (
          <div className="sidebar-section-content">
            {loading ? (
              <div className="sidebar-loading">Loading models...</div>
            ) : error ? (
              <div className="sidebar-error">{error}</div>
            ) : models.length === 0 ? (
              <div className="sidebar-empty">No models available</div>
            ) : (
              <ul className="models-list">
                {models.map((model) => (
                  <li 
                    key={model.id || model.name} 
                    className={`model-item ${activeModel === model.id ? 'active' : ''}`}
                    onClick={() => handleModelSwitch(model.id)}
                  >
                    <SmartToyIcon />
                    <span>{model.name}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-header" onClick={toggleHistory}>
          <div className="sidebar-section-title">
            <HistoryIcon />
            <span>History</span>
          </div>
          {showHistory ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
        </div>
        
        {showHistory && (
          <div className="sidebar-section-content">
            {!chatHistory || chatHistory.length === 0 ? (
              <div className="sidebar-empty">No chat history</div>
            ) : (
              <>
                <ul className="history-list">
                  {chatHistory.map((chat, index) => (
                    <li key={index} className="history-item">
                      <ChatIcon />
                      <span>{chat.message ? chat.message.substring(0, 25) + '...' : 'Chat ' + (index + 1)}</span>
                    </li>
                  ))}
                </ul>
                <button className="clear-history-button" onClick={handleClearHistory}>
                  <DeleteIcon />
                  <span>Clear History</span>
                </button>
              </>
            )}
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <div className="sidebar-section-title">
            <FolderIcon />
            <span>Files</span>
          </div>
        </div>
        <div className="sidebar-section-content">
          <Link to="/files" className="sidebar-link">
            <span>Upload Files</span>
          </Link>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <div className="sidebar-section-title">
            <BarChartIcon />
            <span>Visualizations</span>
          </div>
        </div>
        <div className="sidebar-section-content">
          <Link to="/dashboard/recent" className="sidebar-link">
            <span>Recent Dashboards</span>
          </Link>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar; 