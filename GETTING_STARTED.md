# Getting Started with RR-Chatbot

This guide will help you start and use the RR-Chatbot application as a beginner.

## Starting the Application

### Step 1: Start the Backend Server

1. Open a terminal/command prompt
2. Navigate to the backend directory:
   ```
   cd backend
   ```
3. Activate the virtual environment:
   ```
   # On Windows:
   .\venv_new\Scripts\activate
   
   # On Mac/Linux:
   source venv_new/bin/activate
   ```
4. Start the server:
   ```
   python app.py
   ```
   The server will start on http://localhost:5000

### Step 2: Start the Frontend Server

1. Open a new terminal/command prompt
2. Navigate to the frontend directory:
   ```
   cd frontend
   ```
3. Start the React app:
   ```
   npm start
   ```
   The application will open in your browser at http://localhost:3000

## Logging In as Admin

1. Go to http://localhost:3000/login
2. Use the following credentials:
   - Email: admin@example.com
   - Password: admin123

## Admin Features

As an admin, you have full access to:

1. **User Management**
   - View all users
   - Edit user profiles
   - Delete users
   - Change user roles

2. **Content Management**
   - Manage chat models
   - View all conversations
   - Delete inappropriate content

3. **System Settings**
   - Configure application settings
   - Manage API integrations
   - View system logs

4. **Dashboard Management**
   - Create and edit dashboards
   - Share dashboards with users
   - Export dashboard data

## Basic Usage

1. **Chat Interface**
   - Type messages in the input box at the bottom
   - Upload files using the attachment button
   - Switch between different AI models in the sidebar

2. **File Management**
   - Upload files from the File Upload page
   - View and manage your uploaded files
   - Use files in conversations

3. **Dashboard**
   - Create visualizations of your data
   - Customize charts and graphs
   - Export dashboards as PDF or images

4. **Settings**
   - Change your profile information
   - Toggle dark mode
   - Adjust notification preferences
   - Change your password

## Troubleshooting

If you encounter any issues:

1. Make sure both backend and frontend servers are running
2. Check the console for error messages
3. Ensure your database is properly set up
4. Verify that all required packages are installed

## Need Help?

If you need further assistance, please contact the system administrator or refer to the full documentation. 