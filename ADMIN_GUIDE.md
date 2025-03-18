# Admin Guide for RR-Chatbot

This guide will help you use the admin features of the RR-Chatbot application.

## Accessing the Admin Dashboard

1. **Log in as admin**:
   - Go to http://localhost:3000/login
   - Enter the admin credentials:
     - Email: admin@example.com
     - Password: admin123

2. **Navigate to Admin Dashboard**:
   - After logging in, click on your profile icon in the top-right corner
   - Select "Admin Dashboard" from the dropdown menu
   - You will be taken to the admin dashboard at http://localhost:3000/admin

## Admin Dashboard Overview

The admin dashboard is divided into several tabs:

### 1. Overview Tab

This tab provides a quick summary of your system:
- Total number of users
- Active users
- Total chats
- Total messages
- Total files
- Total dashboards

It also offers quick action buttons for common administrative tasks:
- Add User
- Manage Roles
- View Reports
- System Settings

### 2. User Management Tab

This tab allows you to manage all users in the system:
- View a list of all users
- See user details (username, email, role, last login)
- Edit user information (click the pencil icon)
- Delete users (click the trash icon)

#### To edit a user:
1. Click the pencil icon next to the user
2. Modify the user's information in the dialog
3. Click "Save" to apply changes

#### To delete a user:
1. Click the trash icon next to the user
2. Confirm the deletion in the dialog
3. The user will be permanently removed from the system

### 3. Chat Management Tab

This tab allows you to manage all chat conversations:
- View all conversations in the system
- Delete inappropriate content
- Export chat logs

### 4. File Management Tab

This tab allows you to manage all uploaded files:
- View all files uploaded to the system
- Download files
- Delete files
- View file metadata

### 5. System Settings Tab

This tab allows you to configure system-wide settings:
- API configurations
- Model settings
- Security settings
- Notification settings

## Creating Additional Admin Users

To create another admin user:

1. Open a terminal/command prompt
2. Navigate to the backend directory:
   ```
   cd backend
   ```
3. Activate the virtual environment:
   ```
   # On Windows:
   . ../venv_new/Scripts/activate
   
   # On Mac/Linux:
   source ../venv_new/bin/activate
   ```
4. Run the create_admin.py script:
   ```
   python create_admin.py <username> <email> <password>
   ```
   For example:
   ```
   python create_admin.py superadmin superadmin@example.com superadmin123
   ```

## Best Practices for Admin Users

1. **Regular Backups**: Regularly backup the database to prevent data loss.

2. **User Management**: Review user accounts regularly and remove inactive accounts.

3. **Content Moderation**: Regularly check chat logs for inappropriate content.

4. **Security**: Change admin passwords regularly and use strong passwords.

5. **Updates**: Keep the system updated with the latest security patches.

## Troubleshooting

If you encounter issues with the admin dashboard:

1. Make sure both backend and frontend servers are running
2. Check the browser console for error messages
3. Verify that you have admin privileges
4. Try logging out and logging back in
5. Clear your browser cache

For persistent issues, check the server logs for more detailed error information. 