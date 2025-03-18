import os
import io
import base64
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class DataVizService:
    """Service for handling data visualization requests"""
    
    def __init__(self):
        """Initialize the data visualization service"""
        self.data = None
        self.data_info = None
    
    async def load_data(self, data: Union[str, pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load data into the service
        
        Args:
            data: Data to load (file path, DataFrame, or dictionary)
            
        Returns:
            Dictionary with data info
        """
        try:
            if isinstance(data, str):
                # Load from file
                self.data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                # Use DataFrame directly
                self.data = data
            elif isinstance(data, dict):
                # Convert dictionary to DataFrame
                self.data = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported data format")
            
            # Store data info
            self.data_info = {
                'shape': self.data.shape,
                'columns': self.data.columns.tolist(),
                'dtypes': self.data.dtypes.astype(str).to_dict()
            }
            
            return {
                'status': 'success',
                'message': 'Data loaded successfully',
                'info': self.data_info
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    async def create_visualization(
        self,
        viz_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a visualization
        
        Args:
            viz_type: Type of visualization
            params: Visualization parameters
            
        Returns:
            Dictionary with visualization data
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create visualization based on type
            if viz_type == 'bar':
                sns.barplot(data=self.data, x=params.get('x'), y=params.get('y'))
            elif viz_type == 'line':
                sns.lineplot(data=self.data, x=params.get('x'), y=params.get('y'))
            elif viz_type == 'scatter':
                sns.scatterplot(data=self.data, x=params.get('x'), y=params.get('y'))
            elif viz_type == 'histogram':
                sns.histplot(data=self.data[params.get('column')])
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Add title if provided
            if 'title' in params:
                plt.title(params['title'])
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return {
                'status': 'success',
                'type': 'image',
                'format': 'png',
                'data': img_str
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise
    
    async def get_insights(self) -> Dict[str, Any]:
        """
        Get insights about the loaded data
        
        Returns:
            Dictionary with data insights
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            
            insights = {
                'summary': self.data.describe().to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'correlations': self.data.select_dtypes(include=['number']).corr().to_dict()
            }
            
            return {
                'status': 'success',
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error getting insights: {str(e)}")
            raise
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about the data
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary with query results
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            
            # TODO: Implement natural language query processing
            # For now, return data info
            return {
                'status': 'success',
                'message': 'Query processing not implemented yet',
                'data_info': self.data_info
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def export_dashboard(
        self,
        config: Dict[str, Any],
        format: str = 'html'
    ) -> str:
        """
        Export a dashboard
        
        Args:
            config: Dashboard configuration
            format: Export format ('html' or 'pdf')
            
        Returns:
            Path to the exported file
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            
            if format not in ['html', 'pdf']:
                raise ValueError("Invalid export format")
            
            import tempfile
            import os
            from jinja2 import Environment, FileSystemLoader
            import weasyprint
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Create visualizations
                visualizations = []
                for viz_config in config.get('visualizations', []):
                    result = await self.create_visualization(
                        viz_config['type'],
                        viz_config.get('params', {})
                    )
                    visualizations.append({
                        'title': viz_config.get('title', ''),
                        'description': viz_config.get('description', ''),
                        'image': result['data']
                    })
                
                # Get insights if requested
                insights = None
                if config.get('include_insights', False):
                    insights_result = await self.get_insights()
                    insights = insights_result['insights']
                
                # Create context for template
                context = {
                    'title': config.get('title', 'Dashboard'),
                    'description': config.get('description', ''),
                    'visualizations': visualizations,
                    'insights': insights,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Create HTML
                env = Environment(
                    loader=FileSystemLoader('templates'),
                    autoescape=True
                )
                template = env.get_template('dashboard.html')
                html_content = template.render(**context)
                
                # Save to file
                if format == 'html':
                    output_path = os.path.join(temp_dir, 'dashboard.html')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                else:  # PDF
                    output_path = os.path.join(temp_dir, 'dashboard.pdf')
                    weasyprint.HTML(string=html_content).write_pdf(output_path)
                
                return output_path
                
            except Exception as e:
                logger.error(f"Error exporting dashboard: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            raise 