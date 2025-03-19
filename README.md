# AI Chatbot with Data Analysis

A powerful AI chatbot application with data analysis capabilities, built using modern web technologies. This application combines the power of AI with data analysis to provide intelligent insights and visualizations.

## Features

### Core Functionality
- 🤖 AI-powered chat interface with multiple model support
  - Integration with OpenAI's GPT models
  - Custom model fine-tuning capabilities
  - Context-aware conversations
  - Multi-turn dialogue support
- 📊 Data analysis and visualization capabilities
  - Real-time data processing
  - Interactive visualizations
  - Custom dashboard creation
  - Data export in multiple formats
- 🔒 Secure user authentication and authorization
  - JWT-based authentication
  - Role-based access control
  - Session management
  - Secure password handling
- 🌓 Dark/Light mode support
  - System preference detection
  - Custom theme customization
  - Persistent theme selection
- 📱 Responsive design for all devices
  - Mobile-first approach
  - Adaptive layouts
  - Touch-friendly interfaces
  - Cross-browser compatibility

### Chat Features
- Real-time chat with AI models
  - Streaming responses
  - Typing indicators
  - Message status tracking
  - Error recovery
- Message history and persistence
  - Local storage backup
  - Cloud synchronization
  - Search functionality
  - Conversation export
- Code syntax highlighting
  - Multiple language support
  - Custom theme integration
  - Copy-to-clipboard functionality
  - Line number display
- File upload and analysis
  - Drag-and-drop interface
  - Multiple file format support
  - Progress tracking
  - File validation
- Model selection and configuration
  - Temperature control
  - Token limit settings
  - Response format options
  - Custom prompt templates
- Chat export functionality
  - PDF export
  - Markdown format
  - JSON data export
  - Conversation sharing

### Data Analysis Features
- File upload and processing
  - CSV, Excel, JSON support
  - Data validation
  - Automatic type detection
  - Missing value handling
- Data visualization with Plotly
  - Interactive charts
  - Custom plot configurations
  - Multiple chart types
  - Export capabilities
- Statistical analysis
  - Descriptive statistics
  - Correlation analysis
  - Hypothesis testing
  - Trend analysis
- Machine learning model training
  - Model selection
  - Hyperparameter tuning
  - Performance metrics
  - Model persistence
- Interactive dashboards
  - Drag-and-drop layout
  - Real-time updates
  - Custom widgets
  - Dashboard sharing
- Data export capabilities
  - Multiple format support
  - Batch processing
  - Scheduled exports
  - Custom templates

### User Management
- User registration and login
  - Email verification
  - Password strength requirements
  - Account recovery
  - Session management
- Social authentication
  - Google OAuth
  - GitHub integration
  - Twitter authentication
  - Profile synchronization
- Password reset functionality
  - Secure token generation
  - Email notifications
  - Token expiration
  - Rate limiting
- Email verification
  - Double opt-in
  - Resend capability
  - Email templates
  - Verification status
- User profile management
  - Profile customization
  - Avatar upload
  - Preference settings
  - Activity history
- Role-based access control
  - Role hierarchy
  - Permission management
  - Access logging
  - Role assignment

### Admin Features
- User management dashboard
  - User listing
  - Role management
  - Account status
  - Activity monitoring
- System statistics
  - Usage metrics
  - Performance monitoring
  - Resource utilization
  - Error tracking
- Activity monitoring
  - User actions
  - System events
  - Error logs
  - Audit trails
- Model configuration
  - Model parameters
  - API settings
  - Rate limits
  - Cost tracking
- Usage analytics
  - User engagement
  - Feature usage
  - Performance metrics
  - Cost analysis

## Tech Stack

### Frontend
- **Framework**: React.js (v18+)
  - Functional components
  - React hooks
  - Error boundaries
  - Suspense
- **UI Library**: Material-UI (MUI v5)
  - Custom theming
  - Component customization
  - Responsive design
  - Accessibility features
- **State Management**: React Context API
  - Custom hooks
  - State persistence
  - Performance optimization
  - Error handling
- **Routing**: React Router v6
  - Protected routes
  - Dynamic routing
  - Route guards
  - Navigation history
- **HTTP Client**: Axios
  - Request interceptors
  - Response handling
  - Error management
  - Request caching
- **Data Visualization**: Plotly.js
  - Interactive charts
  - Custom layouts
  - Export options
  - Responsive design
- **Code Highlighting**: Prism.js
  - Multiple languages
  - Custom themes
  - Line numbers
  - Copy functionality
- **Styling**: CSS Modules, Styled Components
  - CSS-in-JS
  - Theme integration
  - Dynamic styles
  - Responsive design

### Backend
- **Framework**: Node.js with Express
  - RESTful API
  - Middleware support
  - Error handling
  - Request validation
- **Database**: MongoDB
  - Schema design
  - Indexing
  - Aggregation
  - Data validation
- **Authentication**: JWT
  - Token management
  - Refresh tokens
  - Token validation
  - Security measures
- **AI Models**: OpenAI API
  - Model selection
  - Parameter tuning
  - Response handling
  - Error management
- **File Storage**: AWS S3
  - File upload
  - Access control
  - Versioning
  - Backup
- **API Documentation**: Swagger/OpenAPI
  - API specification
  - Interactive docs
  - Request/response examples
  - Authentication details

### Development Tools
- **Package Manager**: npm/yarn
  - Dependency management
  - Script automation
  - Version control
  - Package security
- **Version Control**: Git
  - Branch management
  - Code review
  - CI/CD integration
  - Version tagging
- **Code Quality**: ESLint, Prettier
  - Code formatting
  - Style guidelines
  - Error detection
  - Auto-fixing
- **Testing**: Jest, React Testing Library
  - Unit testing
  - Integration testing
  - E2E testing
  - Coverage reporting
- **CI/CD**: GitHub Actions
  - Automated testing
  - Deployment
  - Code quality checks
  - Security scanning

## Project Structure

```
rr-chatbot/
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── assets/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.js
│   │   │   ├── DataAnalysis.js
│   │   │   ├── FileUpload.js
│   │   │   ├── Header.js
│   │   │   ├── Login.js
│   │   │   ├── Register.js
│   │   │   ├── Settings.js
│   │   │   ├── Sidebar.js
│   │   │   └── Visualization.js
│   │   ├── contexts/
│   │   │   ├── AuthContext.js
│   │   │   └── ThemeContext.js
│   │   ├── services/
│   │   │   ├── apiService.js
│   │   │   └── authService.js
│   │   ├── utils/
│   │   │   └── helpers.js
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── README.md
├── backend/
│   ├── config/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── utils/
│   ├── app.js
│   └── package.json
└── README.md
```

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- MongoDB
- OpenAI API key
- AWS account (for S3 storage)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rr-chatbot.git
cd rr-chatbot
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd ../backend
npm install
```

4. Set up environment variables:
Create `.env` files in both frontend and backend directories with the following variables:

Frontend (.env):
```
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_OPENAI_API_KEY=your_openai_api_key
```

Backend (.env):
```
PORT=5000
MONGODB_URI=your_mongodb_uri
JWT_SECRET=your_jwt_secret
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_bucket_name
```

### Running the Application

1. Start the backend server:
```bash
cd backend
npm start
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## API Documentation

The API documentation is available at `/api-docs` when running the backend server. It includes:
- Authentication endpoints
- Chat endpoints
- Data analysis endpoints
- User management endpoints
- Admin endpoints
- File management endpoints

## Development Guidelines

### Code Style
- Follow ESLint and Prettier configurations
- Use functional components and hooks
- Implement proper error handling
- Write meaningful comments
- Follow Git commit conventions

### Testing
- Write unit tests for components
- Implement integration tests
- Use test-driven development
- Maintain test coverage
- Perform regular testing

### Security
- Implement proper authentication
- Use secure password handling
- Validate user input
- Protect sensitive data
- Follow security best practices

### Performance
- Optimize bundle size
- Implement code splitting
- Use proper caching
- Monitor performance
- Optimize images

## Deployment

### Frontend Deployment
1. Build the application:
```bash
npm run build
```

2. Deploy to hosting service:
```bash
npm run deploy
```

### Backend Deployment
1. Set up environment variables
2. Configure database
3. Set up SSL certificates
4. Deploy to server
5. Configure monitoring

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the AI models
- Material-UI for the component library
- Plotly.js for data visualization
- All other open-source libraries used in this project

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Roadmap

### Planned Features
- Voice chat integration
- Advanced data analysis
- Custom model training
- Team collaboration
- API marketplace

### Future Improvements
- Performance optimization
- Enhanced security
- Better user experience
- Additional integrations
- Mobile application

## Changelog

### Version 1.0.0
- Initial release
- Basic chat functionality
- Data analysis features
- User management
- Admin dashboard

### Version 1.1.0 (Planned)
- Enhanced visualization
- Advanced analytics
- Performance improvements
- Bug fixes
- Security updates 