import requests
import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = os.environ.get('API_URL', 'http://localhost:5000')
API_PREFIX = '/api/data-viz'
USERNAME = os.environ.get('TEST_USERNAME', 'admin@example.com')
PASSWORD = os.environ.get('TEST_PASSWORD', 'password')

# Test data
TEST_DATA_DIR = 'test_data'
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def create_test_data():
    """Create test data for API testing"""
    print("Creating test data...")
    
    # Create a sample CSV file
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'sales': np.random.randint(100, 1000, 100),
        'profit': np.random.randint(10, 100, 100),
        'quantity': np.random.randint(1, 50, 100)
    })
    
    # Save to CSV
    csv_path = os.path.join(TEST_DATA_DIR, 'sales_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Created test CSV file: {csv_path}")
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='region', y='sales', data=df)
    plt.title('Sales by Region')
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(TEST_DATA_DIR, 'sales_viz.png')
    plt.savefig(viz_path)
    plt.close()
    print(f"Created test visualization: {viz_path}")
    
    return csv_path, viz_path

def get_auth_token():
    """Get authentication token"""
    print("Getting authentication token...")
    
    auth_url = f"{BASE_URL}/api/auth/login"
    response = requests.post(auth_url, json={
        'email': USERNAME,
        'password': PASSWORD
    })
    
    if response.status_code == 200:
        token = response.json().get('access_token')
        print("Authentication successful")
        return token
    else:
        print(f"Authentication failed: {response.text}")
        return None

def test_health_check():
    """Test health check endpoint"""
    print("\nTesting health check endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/health"
    response = requests.get(url)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert response.json().get('status') == 'ok'
    
    print("Health check test passed")

def test_load_data(token, csv_path):
    """Test load data endpoint"""
    print("\nTesting load data endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/load-data"
    headers = {'Authorization': f'Bearer {token}'}
    
    with open(csv_path, 'rb') as f:
        files = {'file': (os.path.basename(csv_path), f, 'text/csv')}
        data = {'file_type': 'csv'}
        
        response = requests.post(url, headers=headers, files=files, data=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'file_id' in response.json()
    assert 'file_path' in response.json()
    
    print("Load data test passed")
    return response.json().get('file_path')

def test_process_query(token):
    """Test process query endpoint"""
    print("\nTesting process query endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/process-query"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'query': 'Show me sales by region',
        'parameters': {
            'condition': "date > '2023-01-01'"
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'processed_query' in response.json()
    
    print("Process query test passed")

def test_execute_query(token):
    """Test execute query endpoint"""
    print("\nTesting execute query endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/execute-query"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'query': "SELECT region, SUM(sales) FROM data WHERE date > '2023-01-01' GROUP BY region",
        'parameters': {
            'date': '2023-01-01'
        },
        'timeout': 30
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'results' in response.json()
    
    print("Execute query test passed")

def test_visualize(token, file_path):
    """Test visualize endpoint"""
    print("\nTesting visualize endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/visualize"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'data_source': file_path,
        'chart_type': 'bar',
        'title': 'Sales by Region',
        'x_axis': 'region',
        'y_axis': 'sales',
        'filters': {},
        'options': {
            'color': 'blue',
            'orientation': 'vertical'
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'visualization_id' in response.json()
    assert 'visualization_url' in response.json()
    
    print("Visualize test passed")
    return response.json().get('visualization_id')

def test_insights(token, file_path):
    """Test insights endpoint"""
    print("\nTesting insights endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/insights"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'data_source': file_path,
        'insight_type': 'auto',
        'options': {}
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'insights' in response.json()
    
    print("Insights test passed")

def test_preview_data(token, file_path):
    """Test preview data endpoint"""
    print("\nTesting preview data endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/preview-data"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'data_source': file_path,
        'limit': 5,
        'offset': 0,
        'filters': {}
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'records' in response.json()
    assert len(response.json().get('records')) <= 5
    
    print("Preview data test passed")

def test_export_dashboard(token):
    """Test export dashboard endpoint"""
    print("\nTesting export dashboard endpoint...")
    
    url = f"{BASE_URL}{API_PREFIX}/export-dashboard"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'dashboard_id': 'test-dashboard',
        'format': 'pdf',
        'options': {
            'page_size': 'A4',
            'orientation': 'landscape'
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json().get('success') == True
    assert 'export_id' in response.json()
    assert 'export_url' in response.json()
    
    print("Export dashboard test passed")

def run_tests():
    """Run all tests"""
    print("Starting API tests...")
    
    # Create test data
    csv_path, viz_path = create_test_data()
    
    # Get authentication token
    token = get_auth_token()
    if not token:
        print("Cannot proceed with tests without authentication token")
        return
    
    # Run tests
    test_health_check()
    file_path = test_load_data(token, csv_path)
    test_process_query(token)
    test_execute_query(token)
    test_visualize(token, file_path)
    test_insights(token, file_path)
    test_preview_data(token, file_path)
    test_export_dashboard(token)
    
    print("\nAll tests completed successfully!")

if __name__ == '__main__':
    run_tests() 