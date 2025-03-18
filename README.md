# AI Chatbot with Data Visualization

A powerful AI chatbot application with data visualization capabilities, built with Flask, React, and Material UI.

## Features

- **Multiple AI Models**: Supports OLLAMA models, Mistral API, and more
- **Data Visualization**: Upload datasets and generate visualizations directly in the chat
- **File Upload**: Upload and analyze various file types
- **Voice Recording**: Record voice messages for transcription
- **Responsive UI**: Works on desktop and mobile devices
- **Dark Mode**: Toggle between light and dark themes
- **Model Settings**: Adjust temperature, max tokens, and other parameters
- **Error Handling**: Robust error handling with fallbacks
- **SQLite Database**: Store chat history, user data, and logs

## Visualization Features

- **Automatic Dashboard Generation**: Upload a dataset and get a comprehensive dashboard
- **Custom Visualizations**: Generate specific visualizations like histograms, scatter plots, etc.
- **Data Analysis**: Get insights and statistics about your data
- **Interactive UI**: View, download, and share visualizations

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- OLLAMA (optional, for local models)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. Install backend dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```
   cd frontend
   npm install
   cd ..
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   FLASK_ENV=development
   SECRET_KEY=your_secret_key
   MISTRAL_API_KEY=your_mistral_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token
   UPLOAD_FOLDER=uploads
   ```

5. Start the backend server:
   ```
   python app.py
   ```

6. Start the frontend development server:
   ```
   cd frontend
   npm start
   ```

7. Open your browser and navigate to `http://localhost:3000`

## Usage

### Uploading and Visualizing Data

1. Click the file upload button in the chat interface
2. Select a data file (CSV, Excel, JSON, etc.)
3. Ask the chatbot to visualize the data:
   - "Visualize this data"
   - "Create a dashboard for this dataset"
   - "Show me a histogram of column X"
   - "Generate a scatter plot of X vs Y"

### Adjusting Model Settings

1. Click the settings icon in the chat interface
2. Select a different model from the available options
3. Adjust parameters like temperature, max tokens, etc.
4. Continue chatting with the new settings

## API Endpoints

### Chat Endpoints

- `POST /api/simple-chat`: Send a message to the chatbot
- `POST /api/public/chat`: Public chat endpoint (no authentication required)
- `POST /api/chat`: Authenticated chat endpoint

### Visualization Endpoints

- `POST /api/visualization/visualize`: Generate a visualization for a dataset
- `POST /api/visualization/dashboard`: Generate a comprehensive dashboard for a dataset
- `GET /api/visualization/visualizations/<filename>`: Get a visualization image

### File Endpoints

- `POST /api/files/upload`: Upload files
- `GET /api/files/list`: List uploaded files
- `GET /api/files/download/<filename>`: Download a file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [React](https://reactjs.org/)
- [Material UI](https://mui.com/)
- [OLLAMA](https://ollama.ai/)
- [Mistral AI](https://mistral.ai/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/) 