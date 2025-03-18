# Enhanced Data Visualization API

This API provides a comprehensive set of endpoints for data visualization and analysis. It is designed to be integrated into other applications to provide powerful data visualization capabilities.

## Features

- **Robust Error Handling**: Comprehensive error handling with detailed error messages
- **Request Validation**: Schema-based validation for all API requests
- **Authentication**: JWT-based authentication for secure access
- **Rate Limiting**: Prevents abuse by limiting request rates
- **Caching**: Improves performance by caching responses
- **CORS Support**: Enables cross-origin requests
- **Logging**: Detailed logging for monitoring and debugging
- **File Upload**: Secure file upload with validation
- **Data Processing**: Powerful data processing capabilities
- **Visualization Generation**: Multiple visualization types supported
- **Insights Generation**: Automated insights from data
- **Data Preview**: Preview data before visualization
- **Dashboard Export**: Export dashboards in multiple formats

## API Endpoints

### Health Check

```
GET /api/data-viz/health
```

Returns the health status of the API.

**Response:**
```json
{
  "success": true,
  "status": "ok",
  "timestamp": "2023-03-13T12:34:56.789Z",
  "version": "1.1.0"
}
```

### Load Data

```
POST /api/data-viz/load-data
```

Loads data from a file.

**Request:**
```json
{
  "file": <file>,
  "file_type": "csv",
  "options": {
    "delimiter": ",",
    "header": 0
  }
}
```

**Response:**
```json
{
  "success": true,
  "file_id": "123e4567-e89b-12d3-a456-426614174000",
  "file_path": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "file_name": "data.csv",
  "file_type": "csv",
  "stats": {
    "rows": 1000,
    "columns": 10,
    "column_names": ["id", "name", "value", ...],
    "data_types": {
      "id": "int64",
      "name": "object",
      "value": "float64",
      ...
    },
    "missing_values": {
      "id": 0,
      "name": 5,
      "value": 10,
      ...
    }
  }
}
```

### Process Query

```
POST /api/data-viz/process-query
```

Processes a natural language query.

**Request:**
```json
{
  "query": "Show me sales by region",
  "parameters": {
    "condition": "date > '2023-01-01'"
  }
}
```

**Response:**
```json
{
  "success": true,
  "query": "Show me sales by region",
  "processed_query": "SELECT region, SUM(sales) FROM data WHERE date > '2023-01-01' GROUP BY region",
  "parameters": {
    "condition": "date > '2023-01-01'"
  }
}
```

### Execute Query

```
POST /api/data-viz/execute-query
```

Executes a SQL query.

**Request:**
```json
{
  "query": "SELECT region, SUM(sales) FROM data WHERE date > '2023-01-01' GROUP BY region",
  "parameters": {
    "date": "2023-01-01"
  },
  "timeout": 30
}
```

**Response:**
```json
{
  "success": true,
  "query": "SELECT region, SUM(sales) FROM data WHERE date > '2023-01-01' GROUP BY region",
  "parameters": {
    "date": "2023-01-01"
  },
  "results": [
    {"region": "North", "sum": 1000},
    {"region": "South", "sum": 2000},
    {"region": "East", "sum": 1500},
    {"region": "West", "sum": 1800}
  ],
  "execution_time": 0.1
}
```

### Visualize

```
POST /api/data-viz/visualize
```

Generates a visualization.

**Request:**
```json
{
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "chart_type": "bar",
  "title": "Sales by Region",
  "x_axis": "region",
  "y_axis": "sales",
  "filters": {
    "date": "2023-01-01"
  },
  "options": {
    "color": "blue",
    "orientation": "vertical"
  }
}
```

**Response:**
```json
{
  "success": true,
  "visualization_id": "123e4567-e89b-12d3-a456-426614174000",
  "visualization_url": "/visualizations/123e4567-e89b-12d3-a456-426614174000.png",
  "title": "Sales by Region",
  "chart_type": "bar",
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "parameters": {
    "x_axis": "region",
    "y_axis": "sales",
    "filters": {
      "date": "2023-01-01"
    },
    "options": {
      "color": "blue",
      "orientation": "vertical"
    }
  },
  "created_at": "2023-03-13T12:34:56.789Z"
}
```

### Get Insights

```
POST /api/data-viz/insights
```

Gets insights from data.

**Request:**
```json
{
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "insight_type": "correlation",
  "options": {
    "threshold": 0.7
  }
}
```

**Response:**
```json
{
  "success": true,
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "insight_type": "correlation",
  "insights": [
    {
      "type": "correlation",
      "description": "Strong positive correlation (0.85) between sales and marketing",
      "strength": 0.85,
      "columns": ["sales", "marketing"]
    },
    {
      "type": "correlation",
      "description": "Strong negative correlation (-0.75) between price and demand",
      "strength": 0.75,
      "columns": ["price", "demand"]
    }
  ],
  "created_at": "2023-03-13T12:34:56.789Z"
}
```

### Preview Data

```
POST /api/data-viz/preview-data
```

Previews data from a source.

**Request:**
```json
{
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "limit": 10,
  "offset": 0,
  "filters": {
    "region": "North"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data_source": "/uploads/123e4567-e89b-12d3-a456-426614174000_data.csv",
  "records": [
    {"id": 1, "region": "North", "sales": 100, "date": "2023-01-01"},
    {"id": 2, "region": "North", "sales": 150, "date": "2023-01-02"},
    ...
  ],
  "total_count": 250,
  "limit": 10,
  "offset": 0,
  "columns": ["id", "region", "sales", "date"],
  "data_types": {
    "id": "int64",
    "region": "object",
    "sales": "float64",
    "date": "object"
  }
}
```

### Export Dashboard

```
POST /api/data-viz/export-dashboard
```

Exports a dashboard.

**Request:**
```json
{
  "dashboard_id": "123e4567-e89b-12d3-a456-426614174000",
  "format": "pdf",
  "options": {
    "page_size": "A4",
    "orientation": "landscape"
  }
}
```

**Response:**
```json
{
  "success": true,
  "dashboard_id": "123e4567-e89b-12d3-a456-426614174000",
  "export_id": "123e4567-e89b-12d3-a456-426614174000",
  "export_url": "/exports/123e4567-e89b-12d3-a456-426614174000.pdf",
  "format": "pdf",
  "created_at": "2023-03-13T12:34:56.789Z"
}
```

## Authentication

All endpoints except the health check require authentication. Authentication is done using JWT tokens.

To authenticate, include the JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

## Error Handling

All endpoints return a consistent error response format:

```json
{
  "success": false,
  "error": "Error type",
  "message": "Detailed error message",
  "details": {
    "field1": ["Error message for field1"],
    "field2": ["Error message for field2"]
  }
}
```

Common error types:
- Validation error (400)
- Authentication error (401)
- Authorization error (403)
- Not found error (404)
- Rate limit exceeded (429)
- Internal server error (500)

## Rate Limiting

Rate limiting is applied to prevent abuse. The limits are:
- Process Query: 50 requests per minute
- Execute Query: 20 requests per minute
- Other endpoints: 100 requests per minute

## Caching

Responses are cached to improve performance. The cache duration is 1 hour by default.

## CORS Support

CORS is enabled for all endpoints, allowing cross-origin requests.

## Logging

All requests and errors are logged for monitoring and debugging purposes.

## File Upload

Files can be uploaded using the load-data endpoint. The maximum file size is 16MB.

## Data Processing

Data processing is done using pandas and numpy. The API supports CSV, Excel, and JSON file formats.

## Visualization Generation

The API supports multiple visualization types:
- Bar chart
- Line chart
- Pie chart
- Scatter plot
- Heatmap
- Histogram
- Box plot
- Auto (automatically selects the best chart type)

## Insights Generation

The API can generate insights from data:
- Correlation analysis
- Outlier detection
- Trend analysis
- Summary statistics
- Auto (generates all applicable insights)

## Dashboard Export

Dashboards can be exported in multiple formats:
- PDF
- PNG
- HTML
- JSON 