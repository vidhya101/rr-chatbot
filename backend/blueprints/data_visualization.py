from flask import Blueprint, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

data_viz_bp = Blueprint('data_visualization', __name__)

@data_viz_bp.route('/datasets/<dataset_id>/interactive-visualize', methods=['POST'])
def create_interactive_visualization(dataset_id):
    try:
        data = request.get_json()
        chart_type = data.get('chart_type', 'scatter')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        # Load dataset (you'll need to implement your data loading logic)
        # For now, using a dummy dataset
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        # Create visualization based on chart type
        if chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column)
        elif chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_column, y=y_column)
        else:
            return jsonify({'error': 'Unsupported chart type'}), 400
            
        return jsonify({
            'plot_data': json.loads(fig.to_json())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_viz_bp.route('/datasets/<dataset_id>/summary', methods=['GET'])
def get_dataset_summary(dataset_id):
    try:
        # Load dataset (implement your data loading logic)
        # For now, using a dummy dataset
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'summary_stats': json.loads(df.describe().to_json())
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 