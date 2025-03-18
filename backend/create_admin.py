import os
import sys
from flask import Flask
from models.db import db, init_db
from models.user import User
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
init_db(app)

def create_admin_user(username, email, password):
    """Create an admin user"""
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(f"User with email {email} already exists.")
            return
        
        # Create new admin user
        admin_user = User(
            username=username,
            email=email,
            password=password,
            role='admin',
            first_name='Admin',
            last_name='User',
            bio='System administrator with full access',
            is_active=True
        )
        
        # Add to database
        db.session.add(admin_user)
        db.session.commit()
        
        print(f"Admin user '{username}' created successfully!")
        print(f"Email: {email}")
        print(f"Password: {password}")
        print("This user has full admin privileges.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_admin.py <username> <email> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    email = sys.argv[2]
    password = sys.argv[3]
    
    create_admin_user(username, email, password) 