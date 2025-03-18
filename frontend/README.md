# RR-Chatbot Frontend

This is the frontend for the RR-Chatbot application, built with React and Material UI.

## Features

- Modern and responsive UI with Material UI components
- Dark mode support
- Chat interface with multiple AI models
- File upload and processing
- User authentication and profile management
- Dashboard with data visualization
- Settings management

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

2. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
frontend/
├── public/                # Static files
├── src/
│   ├── components/        # React components
│   ├── services/          # API services
│   ├── styles/            # CSS styles
│   ├── utils/             # Utility functions
│   ├── App.js             # Main React component
│   └── index.js           # Entry point
└── package.json           # Node.js dependencies
```

## Available Scripts

- `npm start` - Starts the development server
- `npm build` - Builds the app for production
- `npm test` - Runs tests
- `npm eject` - Ejects from Create React App

## Dependencies

- React
- React Router
- Material UI
- Axios
- Chart.js
- Socket.io Client
- And more (see package.json) 