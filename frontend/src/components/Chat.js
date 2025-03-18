import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  CircularProgress,
  IconButton,
  Divider,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { styled } from '@mui/material/styles';
import chatService from '../services/chatService';
import ModelSelector from './ModelSelector';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Styled components
const ChatContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  padding: theme.spacing(2),
}));

const MessageContainer = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  overflowY: 'auto',
  marginBottom: theme.spacing(2),
  padding: theme.spacing(1),
}));

const MessageBubble = styled(Paper)(({ theme, isUser }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(1),
  maxWidth: '80%',
  borderRadius: 12,
  backgroundColor: isUser ? theme.palette.primary.main : theme.palette.background.paper,
  color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  boxShadow: theme.shadows[1],
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginTop: theme.spacing(1),
}));

const StyledMarkdown = styled(Box)(({ theme }) => ({
  '& p': {
    marginTop: 0,
    marginBottom: theme.spacing(1),
  },
  '& code': {
    backgroundColor: theme.palette.action.hover,
    padding: '2px 4px',
    borderRadius: 4,
    fontFamily: 'monospace',
  },
  '& pre': {
    margin: 0,
    padding: 0,
  },
}));

const Chat = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: message };
    setChatHistory([...chatHistory, userMessage]);
    setMessage('');
    setLoading(true);

    try {
      // Send message to backend
      const response = await chatService.sendMessage({
        message: message,
        chatHistory: chatHistory,
        model: selectedModel
      });
      
      // Add AI response to chat
      const aiMessage = { 
        role: 'assistant', 
        content: response.message,
        chatId: response.chatId,
        model: response.model
      };
      setChatHistory(prev => [...prev, aiMessage]);
      
      // Update current chat ID if this is a new chat
      if (!currentChatId && response.chatId) {
        setCurrentChatId(response.chatId);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message to chat
      const errorMessage = { 
        role: 'assistant', 
        content: error.message || 'Sorry, I encountered an error. Please try again later.' 
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleClearChat = () => {
    setChatHistory([]);
    setCurrentChatId(null);
  };

  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
  };

  // Custom renderer for code blocks
  const renderers = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={atomDark}
          language={match[1]}
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
    }
  };

  return (
    <ChatContainer>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h1">
          Chat with AI
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title="Clear chat">
            <IconButton onClick={handleClearChat} color="primary">
              <DeleteIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <ModelSelector 
        selectedModel={selectedModel} 
        onModelChange={handleModelChange} 
      />
      
      <Divider sx={{ mb: 2 }} />
      
      <MessageContainer>
        {chatHistory.length === 0 ? (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%',
            opacity: 0.7
          }}>
            <Typography variant="body1" color="textSecondary" align="center">
              Start a conversation with the AI assistant.
            </Typography>
            <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 1 }}>
              You can select different AI models using the dropdown above.
            </Typography>
          </Box>
        ) : (
          chatHistory.map((msg, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                mb: 2,
              }}
            >
              <MessageBubble isUser={msg.role === 'user'}>
                {msg.role === 'user' ? (
                  <Typography>{msg.content}</Typography>
                ) : (
                  <StyledMarkdown>
                    <ReactMarkdown components={renderers}>
                      {msg.content}
                    </ReactMarkdown>
                  </StyledMarkdown>
                )}
              </MessageBubble>
            </Box>
          ))
        )}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
            <Paper
              sx={{
                p: 2,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 60,
                height: 40,
              }}
            >
              <CircularProgress size={20} />
            </Paper>
          </Box>
        )}
        <div ref={messagesEndRef} />
      </MessageContainer>

      <form onSubmit={handleSendMessage}>
        <InputContainer>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            multiline
            maxRows={4}
            disabled={loading}
            sx={{ mr: 1 }}
          />
          <Button
            variant="contained"
            color="primary"
            endIcon={<SendIcon />}
            type="submit"
            disabled={!message.trim() || loading}
          >
            Send
          </Button>
        </InputContainer>
      </form>
    </ChatContainer>
  );
};

export default Chat; 