import React, { useState, useRef, useEffect, useCallback } from 'react';
import './ChatInterface.css';

// Material UI components
import { 
  TextField, 
  Button, 
  CircularProgress, 
  IconButton, 
  Tooltip,
  Paper,
  Typography,
  Divider,
  Snackbar,
  Alert,
  Slider,
  Box
} from '@mui/material';

// Material UI Icons
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import TuneIcon from '@mui/icons-material/Tune';
import CloseIcon from '@mui/icons-material/Close';
import NavigationIcon from '@mui/icons-material/Navigation';

// Markdown and code highlighting
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, prism } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Services
import { sendMessage, stopGeneration, listModels } from '../services/apiService';
import { saveChat } from '../services/userService';

// Debounce function for performance
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

const ChatInterface = ({ activeModel, darkMode, onModelChange }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [typingIndicator, setTypingIndicator] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState([
    "What can you help me with today?",
    "Tell me about data analysis capabilities",
    "How do I upload and analyze a dataset?",
    "What AI models are available?"
  ]);
  const [showSettings, setShowSettings] = useState(false);
  const [modelSettings, setModelSettings] = useState({
    temperature: 0.7,
    maxTokens: 2000,
    topP: 0.9,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0
  });
  const [availableModels, setAvailableModels] = useState([]);
  const [ollamaStatus, setOllamaStatus] = useState('unknown');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');
  const [showChatNav, setShowChatNav] = useState(false);
  const [sliderValue, setSliderValue] = useState(0);
  
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);
  const settingsRef = useRef(null);
  const messageRefs = useRef({});

  // Define scrollToBottom with useCallback
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messagesEndRef]);

  // Define toggleChatNav with useCallback
  const toggleChatNav = useCallback(() => {
    setShowChatNav(prev => !prev);
  }, []);

  // Update useEffect dependencies
  useEffect(() => {
    const shouldScrollToBottom = 
      messages.length === 0 || 
      sliderValue === messages.length - 2 || 
      messages[messages.length - 1]?.role === 'user';
    
    if (shouldScrollToBottom) {
      scrollToBottom();
      if (messages.length > 0) {
        setSliderValue(messages.length - 1);
      }
    }
  }, [messages, sliderValue, scrollToBottom]);

  // Add keyboard shortcut for chat navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.altKey && e.key === 'n') {
        e.preventDefault();
        toggleChatNav();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [toggleChatNav]);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await listModels();
        setAvailableModels(response.models || []);
        setOllamaStatus(response.ollama_status || 'unknown');
        
        if (response.ollama_status === 'offline') {
          showSnackbar('Ollama server is offline. Using fallback models.', 'warning');
        }
      } catch (err) {
        console.error('Error fetching models:', err);
        setOllamaStatus('offline');
        showSnackbar('Could not fetch available models. Check your connection.', 'error');
      }
    };
    
    fetchModels();
    
    // Refresh models every 60 seconds
    const intervalId = setInterval(fetchModels, 60000);
    return () => clearInterval(intervalId);
  }, []);

  // Handle click outside settings panel
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (settingsRef.current && !settingsRef.current.contains(event.target)) {
        setShowSettings(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle file upload
  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setUploadedFiles([...uploadedFiles, ...files]);
    
    // Preview files in chat
    files.forEach(file => {
      const newMessage = {
        role: 'user',
        content: `Uploaded file: ${file.name}`,
        timestamp: new Date().toISOString(),
        file: file
      };
      setMessages(prevMessages => [...prevMessages, newMessage]);
    });
    
    // Check if any of the files are data files that can be visualized
    const dataFiles = files.filter(file => {
      const ext = file.name.split('.').pop().toLowerCase();
      return ['csv', 'xlsx', 'xls', 'json', 'txt', 'tsv'].includes(ext);
    });
    
    // Add a helpful message about the uploaded file
    setTypingIndicator(true);
    setTimeout(() => {
      setTypingIndicator(false);
      setMessages(prevMessages => [
        ...prevMessages,
        {
          role: 'assistant',
          content: `I see you've uploaded ${files.length > 1 ? 'some files' : 'a file'}! ${
            dataFiles.length > 0 
              ? "I can analyze or visualize the data in these files. Just ask me to 'visualize this data' or 'create a dashboard'." 
              : "Would you like me to help you with something specific about these files?"
          }`,
          timestamp: new Date().toISOString()
        }
      ]);
    }, 1000);
  };

  // Handle file button click
  const handleFileButtonClick = () => {
    fileInputRef.current.click();
  };

  // Handle voice recording
  const handleVoiceRecording = () => {
    if (!isRecording) {
      // Start recording logic would go here
      setIsRecording(true);
    } else {
      // Stop recording logic would go here
      setIsRecording(false);
      // Process the recording and add it to messages
    }
  };

  // Handle message submission
  const handleSubmit = async (e) => {
    e?.preventDefault();
    
    if (!input.trim() && uploadedFiles.length === 0) return;
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
      files: uploadedFiles.length > 0 ? uploadedFiles : undefined
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsGenerating(true);
    setError(null);
    
    // Show typing indicator
    setTypingIndicator(true);
    
    // Maximum number of retries
    const maxRetries = 2;
    let retryCount = 0;
    let success = false;
    
    while (retryCount <= maxRetries && !success) {
      try {
        // Check if this is a visualization request
        const isVisualizationRequest = 
          input.toLowerCase().includes('visualize') || 
          input.toLowerCase().includes('visualization') ||
          input.toLowerCase().includes('chart') ||
          input.toLowerCase().includes('graph') ||
          input.toLowerCase().includes('plot') ||
          input.toLowerCase().includes('dashboard');
        
        // Check if we have data files
        const dataFiles = uploadedFiles.filter(file => {
          const ext = file.name.split('.').pop().toLowerCase();
          return ['csv', 'xlsx', 'xls', 'json', 'txt', 'tsv'].includes(ext);
        });
        
        let response;
        
        // If it's a visualization request and we have data files
        if (isVisualizationRequest && dataFiles.length > 0) {
          // Create FormData for file upload
          const formData = new FormData();
          
          // Add message data
          const messageData = {
            message: input,
            chatHistory: messages.map(msg => ({
              role: msg.role,
              content: msg.content
            })),
            model: activeModel || 'mistral',
            settings: modelSettings
          };
          
          formData.append('data', JSON.stringify(messageData));
          
          // Add the first data file (we'll only visualize one at a time)
          formData.append('file', dataFiles[0]);
          
          // Send the request with file for visualization
          const fetchResponse = await fetch('/api/simple-chat', {
            method: 'POST',
            body: formData
          });
          
          if (!fetchResponse.ok) {
            throw new Error(`Server responded with ${fetchResponse.status}: ${fetchResponse.statusText}`);
          }
          
          response = await fetchResponse.json();
        } else {
          // Send the message to the API with model settings
          response = await sendMessage({
            message: input,
            chatHistory: messages.map(msg => ({
              role: msg.role,
              content: msg.content
            })),
            files: uploadedFiles,
            model: activeModel || 'mistral',
            settings: modelSettings
          });
        }
        
        // Clear uploaded files after sending
        setUploadedFiles([]);
        
        // Hide typing indicator and add response
        setTypingIndicator(false);
        
        // Check if response has message property or if it's directly the message
        const responseContent = response.message || response;
        
        setMessages(prevMessages => [
          ...prevMessages,
          {
            role: 'assistant',
            content: responseContent,
            timestamp: response.timestamp || new Date().toISOString(),
            isLoading: false
          }
        ]);
        
        // Generate new suggested questions based on the conversation
        generateSuggestedQuestions(responseContent);
        
        // Save the chat to history
        try {
          await saveChat({
            message: input,
            response: responseContent,
            model: activeModel || 'mistral',
            timestamp: new Date().toISOString()
          });
        } catch (err) {
          console.error('Error saving chat:', err);
          // Non-critical error, don't show to user
        }
        
        // Check response time
        const responseTime = response.responseTime || 0;
        if (responseTime > 10) {
          showSnackbar(`Response took ${responseTime.toFixed(1)} seconds. Consider using a different model for faster responses.`, 'info');
        }
        
        // Mark as successful
        success = true;
        
      } catch (err) {
        console.error(`Error sending message (attempt ${retryCount + 1}/${maxRetries + 1}):`, err);
        
        // Increment retry count
        retryCount++;
        
        // If we've reached max retries, show error to user
        if (retryCount > maxRetries) {
          setTypingIndicator(false);
          
          // Check for specific error types
          let errorMessage = "I'm having trouble connecting right now. Could you try again in a moment?";
          
          if (err.message && err.message.includes('timeout')) {
            errorMessage = "The request took too long to complete. Try a simpler question or a different model.";
            showSnackbar("Request timed out. Try a different model or a simpler question.", "error");
          } else if (err.message && err.message.includes('Ollama')) {
            errorMessage = "I couldn't connect to the Ollama server. Please check if it's running correctly.";
            showSnackbar("Ollama server connection failed. Using fallback models.", "warning");
            setOllamaStatus('offline');
          } else {
            showSnackbar("Error sending message. Please try again.", "error");
          }
          
          // Add a friendly error message
          setMessages(prevMessages => [
            ...prevMessages,
            {
              role: 'assistant',
              content: errorMessage,
              timestamp: new Date().toISOString(),
              isError: true
            }
          ]);
          
          setError(errorMessage);
        } else {
          // If we still have retries left, wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 1000));
          // Keep the typing indicator on during retries
        }
      }
    }
    
    // Always clean up regardless of success or failure
    setIsLoading(false);
    setIsGenerating(false);
    
    // Focus back on input field
    setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
  };

  // Generate suggested questions based on conversation
  const generateSuggestedQuestions = (lastResponse) => {
    // Simple logic to generate follow-up questions
    const defaultQuestions = [
      "Tell me more about that",
      "What else can you help me with?",
      "How does this work?",
      "Can you explain in more detail?"
    ];
    
    // Extract potential topics from the last response
    const topics = lastResponse
      .split(/[.!?]/)
      .filter(sentence => sentence.length > 20)
      .map(sentence => sentence.trim())
      .slice(0, 2);
    
    // Generate questions based on topics
    const topicQuestions = topics.map(topic => {
      const firstWords = topic.split(' ').slice(0, 3).join(' ');
      return `Tell me more about ${firstWords}...`;
    });
    
    // Combine with some default questions
    setSuggestedQuestions([
      ...topicQuestions,
      ...defaultQuestions.slice(0, 4 - topicQuestions.length)
    ]);
  };

  // Handle suggested question click
  const handleSuggestedQuestionClick = (question) => {
    setInput(question);
    handleSubmit();
  };

  // Handle stopping generation
  const handleStopGeneration = async () => {
    try {
      await stopGeneration();
      setIsGenerating(false);
      setTypingIndicator(false);
    } catch (err) {
      console.error('Error stopping generation:', err);
    }
  };

  // Show snackbar notification
  const showSnackbar = (message, severity = 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  // Handle snackbar close
  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  // Handle model settings change
  const handleSettingChange = useCallback(debounce((setting, value) => {
    setModelSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  }, 300), []);

  // Custom renderer for code blocks in markdown
  const CodeBlock = ({ node, inline, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || '');
    const language = match ? match[1] : '';
    
    return !inline ? (
      <SyntaxHighlighter
        style={darkMode ? vscDarkPlus : prism}
        language={language}
        PreTag="div"
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    );
  };

  // Render typing indicator
  const renderTypingIndicator = () => (
    <Paper elevation={1} className={`message assistant typing-indicator ${darkMode ? 'dark-mode' : ''}`}>
      <div className="message-header">
        <Typography variant="subtitle2">
          {activeModel || 'AI'}
        </Typography>
        <Typography variant="caption">
          {new Date().toLocaleTimeString()}
        </Typography>
      </div>
      
      <Divider />
      
      <div className="message-content">
        <div className="typing-animation">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </Paper>
  );

  // Render settings panel
  const renderSettingsPanel = () => (
    <div className={`settings-panel ${darkMode ? 'dark-mode' : ''}`} ref={settingsRef}>
      <div className="settings-header">
        <Typography variant="h6">Model Settings</Typography>
        <IconButton onClick={() => setShowSettings(false)}>
          <CloseIcon />
        </IconButton>
      </div>
      
      <Divider />
      
      <div className="settings-content">
        <Typography variant="subtitle1" gutterBottom>Available Models</Typography>
        <div className="model-list">
          {availableModels.map(model => (
            <Button
              key={model.id}
              variant={model.id === activeModel ? "contained" : "outlined"}
              size="small"
              onClick={() => handleModelChange(model.id)}
              className="model-button"
              color={model.provider === 'ollama' && ollamaStatus === 'offline' ? 'error' : 'primary'}
              disabled={model.provider === 'ollama' && ollamaStatus === 'offline'}
            >
              {model.name}
              {model.provider === 'ollama' && ollamaStatus === 'offline' && " (Offline)"}
            </Button>
          ))}
        </div>
        
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>Temperature</Typography>
        <Box sx={{ px: 1 }}>
          <Slider
            value={modelSettings.temperature}
            min={0}
            max={1}
            step={0.1}
            marks
            valueLabelDisplay="auto"
            onChange={(e, value) => handleSettingChange('temperature', value)}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          Lower values make responses more focused and deterministic. Higher values make responses more creative and varied.
        </Typography>
        
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>Max Tokens</Typography>
        <Box sx={{ px: 1 }}>
          <Slider
            value={modelSettings.maxTokens}
            min={100}
            max={4000}
            step={100}
            marks
            valueLabelDisplay="auto"
            onChange={(e, value) => handleSettingChange('maxTokens', value)}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          Maximum number of tokens to generate. Higher values allow for longer responses.
        </Typography>
        
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>Top P</Typography>
        <Box sx={{ px: 1 }}>
          <Slider
            value={modelSettings.topP}
            min={0.1}
            max={1}
            step={0.1}
            marks
            valueLabelDisplay="auto"
            onChange={(e, value) => handleSettingChange('topP', value)}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          Controls diversity via nucleus sampling. Lower values make responses more focused.
        </Typography>
      </div>
    </div>
  );

  // Handle model change
  const handleModelChange = (modelId) => {
    if (typeof onModelChange === 'function') {
      onModelChange(modelId);
    } else {
      // If onModelChange prop is not provided, handle it internally
      showSnackbar(`Switched to model: ${modelId}`, 'info');
      // You might want to store the selected model in local storage
      localStorage.setItem('preferredModel', modelId);
    }
    setShowSettings(false);
  };

  // Handle copying message to clipboard
  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content);
    showSnackbar("Copied to clipboard!", "success");
  };

  // Handle feedback (thumbs up/down)
  const handleFeedback = (messageIndex, isPositive) => {
    // Show acknowledgment
    showSnackbar(isPositive ? 'Thanks for the positive feedback!' : 'Thanks for the feedback. We\'ll work to improve.', 'success');
    
    // Mark the message as having received feedback
    setMessages(prevMessages => {
      const newMessages = [...prevMessages];
      newMessages[messageIndex] = {
        ...newMessages[messageIndex],
        feedback: isPositive ? 'positive' : 'negative'
      };
      return newMessages;
    });
    
    // Send feedback to backend
    try {
      const feedbackData = {
        messageId: messages[messageIndex].id || String(messageIndex),
        type: isPositive ? 'positive' : 'negative',
        chatId: messages[messageIndex].chatId,
        content: messages[messageIndex].content,
        model: activeModel
      };
      
      // Use the api instance from apiService
      import('../services/apiService').then(({ default: api }) => {
        api.post('/feedback', feedbackData)
          .catch(err => {
            console.error('Error sending feedback:', err);
            // Non-critical error, don't show to user
          });
      });
    } catch (err) {
      console.error('Error preparing feedback:', err);
      // Non-critical error, don't show to user
    }
  };

  // Scroll to a specific message
  const scrollToMessage = (index) => {
    if (messageRefs.current[index]) {
      messageRefs.current[index].scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
      
      // Add a highlight effect to the message
      const message = messageRefs.current[index];
      message.style.transition = 'box-shadow 0.3s ease, transform 0.3s ease';
      message.style.boxShadow = '0 0 0 2px #2196f3';
      message.style.transform = 'scale(1.02)';
      
      setTimeout(() => {
        message.style.boxShadow = '';
        message.style.transform = '';
      }, 1000);
    }
  };
  
  // Handle slider change
  const handleSliderChange = (event, newValue) => {
    // Only scroll if the value has actually changed
    if (newValue !== sliderValue) {
      setSliderValue(newValue);
      
      // Use requestAnimationFrame for smoother scrolling
      requestAnimationFrame(() => {
        scrollToMessage(newValue);
      });
      
      // Remove any haptic feedback to avoid issues
      // if (navigator.vibrate) {
      //   navigator.vibrate(10);
      // }
    }
  };
  
  // Render chat navigation sidebar
  const renderChatNavSidebar = () => (
    <>
      <div 
        className={`chat-nav-overlay ${showChatNav ? 'visible' : ''}`} 
        onClick={toggleChatNav}
      />
      <div className={`chat-nav-sidebar ${showChatNav ? 'open' : ''}`}>
        <div className="chat-nav-header">
          <Typography variant="subtitle1">Chat Navigation</Typography>
          <IconButton size="small" onClick={toggleChatNav}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </div>
        <Divider />
        <div className="chat-nav-content">
          <Typography variant="body2" gutterBottom>
            Slide to navigate through messages
          </Typography>
          <Box sx={{ px: 2, py: 1 }}>
            <Slider
              value={sliderValue}
              onChange={handleSliderChange}
              min={0}
              max={Math.max(0, messages.length - 1)}
              step={1}
              marks
              valueLabelDisplay="off"
              disabled={messages.length <= 1}
            />
          </Box>
          <div className="chat-nav-message-list">
            {messages.map((message, index) => (
              <div 
                key={index}
                className={`chat-nav-message-item ${sliderValue === index ? 'active' : ''}`}
                onClick={() => {
                  setSliderValue(index);
                  scrollToMessage(index);
                }}
              >
                <Typography variant="caption">
                  {message.role === 'user' ? 'You' : activeModel || 'AI'} - 
                  {new Date(message.timestamp).toLocaleTimeString()}
                </Typography>
                <Typography variant="body2" noWrap>
                  {message.content.substring(0, 50)}
                  {message.content.length > 50 ? '...' : ''}
                </Typography>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <div className={`chat-interface ${darkMode ? 'dark-mode' : ''}`}>
      {renderChatNavSidebar()}
      
      <div className="chat-messages" ref={messagesContainerRef}>
        {messages.length === 0 && !typingIndicator ? (
          <div className="empty-chat">
            <Typography variant="h5" gutterBottom>
              Start a conversation with {activeModel || 'AI'}
            </Typography>
            <Typography variant="body1">
              Ask a question, request information, or start a discussion.
            </Typography>
            
            <div className="suggested-questions">
              {suggestedQuestions.map((question, index) => (
                <Button 
                  key={index}
                  variant="outlined" 
                  size="small"
                  onClick={() => {
                    setInput(question);
                    setTimeout(() => handleSubmit(), 100);
                  }}
                  className="suggested-question"
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <Paper 
                key={index} 
                elevation={1} 
                className={`message ${message.role} ${message.isLoading ? 'loading' : ''} ${message.isError ? 'error' : ''} ${message.feedback ? `feedback-${message.feedback}` : ''}`}
                ref={el => messageRefs.current[index] = el}
              >
                <div className="message-header">
                  <Typography variant="subtitle2">
                    {message.role === 'user' ? 'You' : activeModel || 'AI'}
                  </Typography>
                  <Typography variant="caption">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </Typography>
                </div>
                
                <Divider />
                
                <div className="message-content">
                  {message.isLoading ? (
                    <CircularProgress size={20} />
                  ) : (
                    <ReactMarkdown
                      components={{
                        code: CodeBlock
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  )}
                </div>
                
                {message.role === 'assistant' && !message.isLoading && (
                  <div className="message-actions">
                    <Tooltip title="Copy">
                      <IconButton size="small" onClick={() => handleCopyMessage(message.content)}>
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    
                    <Tooltip title="Helpful">
                      <IconButton 
                        size="small" 
                        onClick={() => handleFeedback(index, true)}
                        color={message.feedback === 'positive' ? 'primary' : 'default'}
                      >
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    
                    <Tooltip title="Not Helpful">
                      <IconButton 
                        size="small" 
                        onClick={() => handleFeedback(index, false)}
                        color={message.feedback === 'negative' ? 'error' : 'default'}
                      >
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    
                    <Tooltip title="More Options">
                      <IconButton size="small">
                        <MoreVertIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </div>
                )}
              </Paper>
            ))}
            
            {typingIndicator && renderTypingIndicator()}
            
            {/* Suggested follow-up questions after the last assistant message */}
            {messages.length > 0 && messages[messages.length - 1].role === 'assistant' && !typingIndicator && (
              <div className="suggested-questions">
                {suggestedQuestions.map((question, index) => (
                  <Button 
                    key={index}
                    variant="outlined" 
                    size="small"
                    onClick={() => handleSuggestedQuestionClick(question)}
                    className="suggested-question"
                  >
                    {question}
                  </Button>
                ))}
              </div>
            )}
          </>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {error && (
        <div className={`error-message ${error.includes('Thanks') ? 'success' : ''}`}>
          {error}
        </div>
      )}
      
      {messages.length > 1 && (
        <div className="message-slider-container">
          <Box sx={{ width: '100%', padding: '0 16px' }}>
            <Typography variant="caption" color="textSecondary">
              Navigate Messages ({sliderValue + 1}/{messages.length})
            </Typography>
            <Slider
              value={sliderValue}
              onChange={handleSliderChange}
              min={0}
              max={messages.length - 1}
              step={1}
              marks={messages.length <= 10 ? Array.from({ length: messages.length }, (_, i) => ({ value: i })) : false}
              valueLabelDisplay="off"
              size="small"
              aria-label="Message navigation"
            />
          </Box>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="chat-input-container">
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileUpload}
          multiple
        />
        
        <IconButton 
          onClick={handleFileButtonClick}
          disabled={isLoading}
          color="primary"
        >
          <AttachFileIcon />
        </IconButton>
        
        <IconButton 
          onClick={handleVoiceRecording}
          disabled={isLoading}
          color={isRecording ? 'error' : 'primary'}
        >
          <MicIcon />
        </IconButton>
        
        <TextField
          className="chat-input"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          variant="outlined"
          fullWidth
          inputRef={inputRef}
          InputProps={{
            endAdornment: isGenerating ? (
              <IconButton onClick={handleStopGeneration} color="error">
                <StopIcon />
              </IconButton>
            ) : null
          }}
        />
        
        <Button
          variant="contained"
          color="primary"
          endIcon={<SendIcon />}
          onClick={handleSubmit}
          disabled={isLoading || (!input.trim() && uploadedFiles.length === 0)}
          className="send-button"
        >
          {isLoading ? <CircularProgress size={24} /> : 'Send'}
        </Button>
        
        <Tooltip title="Chat Navigation (Alt+N)">
          <IconButton
            onClick={toggleChatNav}
            color={showChatNav ? "secondary" : "primary"}
            className={`chat-nav-button ${showChatNav ? 'active' : ''}`}
          >
            <NavigationIcon />
          </IconButton>
        </Tooltip>
        
        <IconButton
          onClick={() => setShowSettings(!showSettings)}
          color="primary"
          className="settings-button"
        >
          <TuneIcon />
        </IconButton>
      </form>
      
      {showSettings && renderSettingsPanel()}
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default ChatInterface; 