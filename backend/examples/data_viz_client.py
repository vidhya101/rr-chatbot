import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataVizClient:
    """Client for the Data Visualization API"""
    
    def __init__(self, base_url=None, api_prefix=None):
        """Initialize the client"""
        self.base_url = base_url or os.environ.get('API_URL', 'http://localhost:5000')
        self.api_prefix = api_prefix or '/api/data-viz'
        self.token = None
    
    def login(self, email, password):
        """Login to get authentication token"""
        auth_url = f"{self.base_url}/api/auth/login"
        response = requests.post(auth_url, json={
            'email': email,
            'password': password
        })
        
        if response.status_code == 200:
            self.token = response.json().get('access_token')
            return True
        else:
            print(f"Authentication failed: {response.text}")
            return False
    
    def get_headers(self):
        """Get headers with authentication token"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        return headers
    
    def health_check(self):
        """Check API health"""
        url = f"{self.base_url}{self.api_prefix}/health"
        response = requests.get(url)
        
        return response.json()
    
    def load_data(self, file_path, file_type='csv', options=None):
        """Load data from a file"""
        url = f"{self.base_url}{self.api_prefix}/load-data"
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, f'text/{file_type}')}
            data = {'file_type': file_type}
            
            if options:
                data['options'] = json.dumps(options)
            
            response = requests.post(
                url,
                headers={'Authorization': f'Bearer {self.token}'},
                files=files,
                data=data
            )
        
        return response.json()
    
    def process_query(self, query, parameters=None):
        """Process a natural language query"""
        url = f"{self.base_url}{self.api_prefix}/process-query"
        
        data = {
            'query': query
        }
        
        if parameters:
            data['parameters'] = parameters
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def execute_query(self, query, parameters=None, timeout=30):
        """Execute a SQL query"""
        url = f"{self.base_url}{self.api_prefix}/execute-query"
        
        data = {
            'query': query,
            'timeout': timeout
        }
        
        if parameters:
            data['parameters'] = parameters
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def visualize(self, data_source, chart_type='bar', title=None, x_axis=None, y_axis=None, filters=None, options=None):
        """Generate a visualization"""
        url = f"{self.base_url}{self.api_prefix}/visualize"
        
        data = {
            'data_source': data_source,
            'chart_type': chart_type
        }
        
        if title:
            data['title'] = title
        
        if x_axis:
            data['x_axis'] = x_axis
        
        if y_axis:
            data['y_axis'] = y_axis
        
        if filters:
            data['filters'] = filters
        
        if options:
            data['options'] = options
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def get_insights(self, data_source, insight_type='auto', options=None):
        """Get insights from data"""
        url = f"{self.base_url}{self.api_prefix}/insights"
        
        data = {
            'data_source': data_source,
            'insight_type': insight_type
        }
        
        if options:
            data['options'] = options
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def preview_data(self, data_source, limit=100, offset=0, filters=None):
        """Preview data from a source"""
        url = f"{self.base_url}{self.api_prefix}/preview-data"
        
        data = {
            'data_source': data_source,
            'limit': limit,
            'offset': offset
        }
        
        if filters:
            data['filters'] = filters
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def export_dashboard(self, dashboard_id, export_format='pdf', options=None):
        """Export a dashboard"""
        url = f"{self.base_url}{self.api_prefix}/export-dashboard"
        
        data = {
            'dashboard_id': dashboard_id,
            'format': export_format
        }
        
        if options:
            data['options'] = options
        
        response = requests.post(
            url,
            headers=self.get_headers(),
            json=data
        )
        
        return response.json()
    
    def get_visualization_image(self, visualization_url):
        """Get a visualization image"""
        url = f"{self.base_url}{visualization_url}"
        
        response = requests.get(
            url,
            headers={'Authorization': f'Bearer {self.token}'}
        )
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to get visualization: {response.text}")
            return None
    
    def display_visualization(self, visualization_result):
        """Display a visualization"""
        if visualization_result.get('success'):
            viz_url = visualization_result.get('visualization_url')
            img = self.get_visualization_image(viz_url)
            
            if img:
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(visualization_result.get('title', 'Visualization'))
                plt.show()
            else:
                print("Failed to display visualization")
        else:
            print(f"Visualization failed: {visualization_result.get('error')}")
    
    def create_dashboard(self, data_source, title="Dashboard"):
        """Create a simple dashboard with multiple visualizations"""
        # Preview the data
        preview = self.preview_data(data_source, limit=5)
        
        if not preview.get('success'):
            print(f"Failed to preview data: {preview.get('error')}")
            return None
        
        # Get insights
        insights = self.get_insights(data_source)
        
        if not insights.get('success'):
            print(f"Failed to get insights: {insights.get('error')}")
            return None
        
        # Create visualizations
        visualizations = []
        
        # Bar chart
        bar_chart = self.visualize(
            data_source=data_source,
            chart_type='bar',
            title='Bar Chart',
            x_axis=preview.get('columns', [])[0] if preview.get('columns') else None,
            y_axis=preview.get('columns', [])[1] if len(preview.get('columns', [])) > 1 else None
        )
        
        if bar_chart.get('success'):
            visualizations.append(bar_chart)
        
        # Line chart
        line_chart = self.visualize(
            data_source=data_source,
            chart_type='line',
            title='Line Chart',
            x_axis=preview.get('columns', [])[0] if preview.get('columns') else None,
            y_axis=preview.get('columns', [])[1] if len(preview.get('columns', [])) > 1 else None
        )
        
        if line_chart.get('success'):
            visualizations.append(line_chart)
        
        # Pie chart
        pie_chart = self.visualize(
            data_source=data_source,
            chart_type='pie',
            title='Pie Chart',
            x_axis=preview.get('columns', [])[0] if preview.get('columns') else None,
            y_axis=preview.get('columns', [])[1] if len(preview.get('columns', [])) > 1 else None
        )
        
        if pie_chart.get('success'):
            visualizations.append(pie_chart)
        
        # Create dashboard
        dashboard = {
            'title': title,
            'data_source': data_source,
            'visualizations': visualizations,
            'insights': insights.get('insights', []),
            'preview': preview.get('records', [])
        }
        
        return dashboard
    
    def display_dashboard(self, dashboard):
        """Display a dashboard"""
        if not dashboard:
            print("No dashboard to display")
            return
        
        print(f"\n=== {dashboard.get('title', 'Dashboard')} ===\n")
        
        # Display preview
        print("Data Preview:")
        if dashboard.get('preview'):
            df = pd.DataFrame(dashboard.get('preview'))
            print(df.head())
        else:
            print("No preview data available")
        
        print("\nInsights:")
        for insight in dashboard.get('insights', []):
            print(f"- {insight.get('description', 'No description')}")
        
        print("\nVisualizations:")
        for viz in dashboard.get('visualizations', []):
            print(f"- {viz.get('title', 'Visualization')}")
            self.display_visualization(viz)


# Example usage
def main():
    """Example usage of the Data Visualization API client"""
    # Create client
    client = DataVizClient()
    
    # Check API health
    health = client.health_check()
    print(f"API Health: {health}")
    
    # Login
    email = os.environ.get('TEST_USERNAME', 'admin@example.com')
    password = os.environ.get('TEST_PASSWORD', 'password')
    
    if not client.login(email, password):
        print("Login failed. Exiting.")
        return
    
    # Create test data
    print("Creating test data...")
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'region': ['North', 'South', 'East', 'West'] * 25,
        'product': ['A', 'B', 'C', 'D'] * 25,
        'sales': [100, 200, 150, 300] * 25,
        'profit': [10, 20, 15, 30] * 25
    })
    
    # Save to CSV
    csv_path = 'example_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Created test CSV file: {csv_path}")
    
    # Load data
    result = client.load_data(csv_path)
    
    if not result.get('success'):
        print(f"Failed to load data: {result.get('error')}")
        return
    
    data_source = result.get('file_path')
    print(f"Data loaded: {data_source}")
    
    # Create dashboard
    dashboard = client.create_dashboard(data_source, "Sales Dashboard")
    
    # Display dashboard
    client.display_dashboard(dashboard)
    
    # Export dashboard
    export_result = client.export_dashboard(
        dashboard_id='example-dashboard',
        export_format='pdf',
        options={
            'page_size': 'A4',
            'orientation': 'landscape'
        }
    )
    
    print(f"Dashboard exported: {export_result}")


if __name__ == '__main__':
    main() 