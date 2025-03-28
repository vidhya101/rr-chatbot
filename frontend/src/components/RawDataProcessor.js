import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  TextField,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Tooltip,
  Divider
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  Visibility as ViewIcon,
  Error as ErrorIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

// Constants
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_FILE_TYPES = ['text/plain', 'text/csv', 'application/json'];
const MAX_PREVIEW_ROWS = 5;

const RawDataProcessor = () => {
  // State management
  const [activeStep, setActiveStep] = useState(0);
  const [rawData, setRawData] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [jsonData, setJsonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [validationErrors, setValidationErrors] = useState([]);
  const fileInputRef = useRef(null);

  // Steps configuration
  const steps = [
    { label: 'Upload Raw Data', description: 'Upload your data file' },
    { label: 'Convert to CSV', description: 'Convert raw data to CSV format' },
    { label: 'Convert to JSON', description: 'Convert CSV to JSON format' },
    { label: 'Analyze with Ollama', description: 'Analyze data structure' },
    { label: 'Generate Dashboard', description: 'Create interactive dashboard' }
  ];

  // Validation functions
  const validateFile = (file) => {
    const errors = [];
    
    if (!file) {
      errors.push('No file selected');
      return errors;
    }

    if (file.size > MAX_FILE_SIZE) {
      errors.push(`File size exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit`);
    }

    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      errors.push('Invalid file type. Please upload a text, CSV, or JSON file');
    }

    return errors;
  };

  // Error handling utility
  const handleError = (error, customMessage = '') => {
    console.error('Error:', error);
    setError(error.message || customMessage);
    setSnackbar({
      open: true,
      message: customMessage || error.message || 'An error occurred',
      severity: 'error'
    });
  };

  // Success handling utility
  const handleSuccess = (message) => {
    setSnackbar({
      open: true,
      message,
      severity: 'success'
    });
  };

  // File upload handler
  const handleRawDataUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      // Validate file
      const errors = validateFile(file);
      if (errors.length > 0) {
        setValidationErrors(errors);
        handleError(new Error(errors.join(', ')));
        return;
      }

      setLoading(true);
      setError(null);
      setValidationErrors([]);

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          setRawData(e.target.result);
          setActiveStep(1);
          handleSuccess('Raw data uploaded successfully');
        } catch (err) {
          handleError(err, 'Error reading file');
        } finally {
          setLoading(false);
        }
      };

      reader.onerror = () => {
        handleError(new Error('Error reading file'), 'Failed to read file');
        setLoading(false);
      };

      reader.readAsText(file);
    } catch (err) {
      handleError(err, 'Error processing file');
      setLoading(false);
    }
  };

  // CSV conversion handler
  const convertToCSV = () => {
    try {
      if (!rawData) {
        throw new Error('No raw data available');
      }

      setLoading(true);
      const lines = rawData.split('\n').filter(line => line.trim());
      
      if (lines.length === 0) {
        throw new Error('Empty file');
      }

      const result = lines.map(line => 
        line.split(/[,\t]/).map(cell => cell.trim())
      );

      // Validate CSV structure
      const headerLength = result[0].length;
      const invalidRows = result.filter(row => row.length !== headerLength);
      
      if (invalidRows.length > 0) {
        throw new Error(`Invalid CSV structure: ${invalidRows.length} rows have incorrect number of columns`);
      }

      setCsvData(result);
      setActiveStep(2);
      handleSuccess('Converted to CSV successfully');
    } catch (err) {
      handleError(err, 'Error converting to CSV');
    } finally {
      setLoading(false);
    }
  };

  // JSON conversion handler
  const convertToJSON = () => {
    try {
      if (!csvData || csvData.length < 2) {
        throw new Error('Invalid CSV data');
      }

      setLoading(true);
      const headers = csvData[0];
      const jsonResult = csvData.slice(1).map(row => {
        const obj = {};
        headers.forEach((header, index) => {
          // Handle numeric values
          const value = row[index];
          obj[header] = !isNaN(value) ? parseFloat(value) : value;
        });
        return obj;
      });

      setJsonData(jsonResult);
      setActiveStep(3);
      handleSuccess('Converted to JSON successfully');
    } catch (err) {
      handleError(err, 'Error converting to JSON');
    } finally {
      setLoading(false);
    }
  };

  // Download handlers
  const downloadJSON = () => {
    try {
      if (!jsonData) {
        throw new Error('No JSON data available');
      }

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'converted_data.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      handleSuccess('JSON file downloaded successfully');
    } catch (err) {
      handleError(err, 'Error downloading JSON file');
    }
  };

  // Dashboard generation
  const downloadDashboard = () => {
    try {
      if (!jsonData) {
        throw new Error('No JSON data available for dashboard');
      }

      const dashboardHTML = generateDashboardHTML();
      const blob = new Blob([dashboardHTML], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sales_dashboard.html';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      handleSuccess('Dashboard file downloaded successfully');
    } catch (err) {
      handleError(err, 'Error generating dashboard');
    }
  };

  // Data preview renderer
  const renderDataPreview = (data) => {
    if (!data) return null;

    if (typeof data === 'string') {
      return (
        <TextField
          multiline
          fullWidth
          rows={10}
          value={data}
          variant="outlined"
          InputProps={{ readOnly: true }}
          error={!!error}
          helperText={error}
        />
      );
    }

    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {Object.keys(data[0] || {}).map((header) => (
                <TableCell key={header}>{header}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.slice(0, MAX_PREVIEW_ROWS).map((row, index) => (
              <TableRow key={index}>
                {Object.values(row).map((cell, cellIndex) => (
                  <TableCell key={cellIndex}>{cell}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  // Current step renderer
  const renderCurrentStep = () => {
    if (loading) {
      return (
        <Box display="flex" justifyContent="center" p={3}>
          <CircularProgress />
        </Box>
      );
    }

    switch (activeStep) {
      case 0:
        return (
          <Card>
            <CardContent>
              <Box textAlign="center" p={3}>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={handleRawDataUpload}
                  accept={ALLOWED_FILE_TYPES.join(',')}
                />
                <Button
                  variant="contained"
                  startIcon={<UploadIcon />}
                  onClick={() => fileInputRef.current.click()}
                  disabled={loading}
                >
                  Upload Raw Data
                </Button>
                {validationErrors.length > 0 && (
                  <Box mt={2}>
                    {validationErrors.map((error, index) => (
                      <Alert key={index} severity="error" sx={{ mb: 1 }}>
                        {error}
                      </Alert>
                    ))}
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        );

      case 1:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Raw Data Preview
              </Typography>
              {renderDataPreview(rawData)}
              <Box display="flex" justifyContent="flex-end" mt={2}>
                <Button
                  variant="contained"
                  onClick={convertToCSV}
                  startIcon={<RefreshIcon />}
                  disabled={loading}
                >
                  Convert to CSV
                </Button>
              </Box>
            </CardContent>
          </Card>
        );

      case 2:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CSV Data Preview
              </Typography>
              {renderDataPreview(csvData)}
              <Box display="flex" justifyContent="flex-end" mt={2}>
                <Button
                  variant="contained"
                  onClick={convertToJSON}
                  startIcon={<RefreshIcon />}
                  disabled={loading}
                >
                  Convert to JSON
                </Button>
              </Box>
            </CardContent>
          </Card>
        );

      case 3:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                JSON Data Preview
              </Typography>
              {renderDataPreview(jsonData)}
              <Box display="flex" justifyContent="flex-end" mt={2} gap={2}>
                <Button
                  variant="contained"
                  onClick={downloadJSON}
                  startIcon={<DownloadIcon />}
                  disabled={loading}
                >
                  Download JSON
                </Button>
                <Button
                  variant="contained"
                  onClick={downloadDashboard}
                  startIcon={<SaveIcon />}
                  color="secondary"
                  disabled={loading}
                >
                  Download Dashboard
                </Button>
              </Box>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  // Snackbar handler
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Dashboard HTML generator
  const generateDashboardHTML = () => {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Sales Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5; 
        }
        .dashboard { 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .header { 
            background: #fff; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 20px; 
        }
        .chart { 
            background: #fff; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            min-height: 400px;
        }
        .chart.full-width {
            grid-column: 1 / -1;
        }
        .upload-section { 
            text-align: center; 
            margin-bottom: 20px; 
        }
        #fileInput { 
            display: none; 
        }
        .upload-btn { 
            background: #2196F3; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s ease;
        }
        .upload-btn:hover { 
            background: #1976D2; 
        }
        .error-message {
            color: #d32f2f;
            margin-top: 10px;
            text-align: center;
        }
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        }
        .loading::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2196F3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            justify-content: center;
        }
        .control-btn {
            background: #fff;
            border: 1px solid #2196F3;
            color: #2196F3;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .control-btn:hover {
            background: #2196F3;
            color: white;
        }
        .control-btn.active {
            background: #2196F3;
            color: white;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Advanced Sales Analytics Dashboard</h1>
            <div class="upload-section">
                <input type="file" id="fileInput" accept=".json" />
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">Upload JSON Data</button>
            </div>
            <div class="controls">
                <button class="control-btn active" onclick="toggleView('overview')">Overview</button>
                <button class="control-btn" onclick="toggleView('trends')">Trends</button>
                <button class="control-btn" onclick="toggleView('comparison')">Comparison</button>
                <button class="control-btn" onclick="toggleView('predictions')">Predictions</button>
            </div>
        </div>
        <div class="grid">
            <!-- Overview Section -->
            <div class="chart" id="salesTrend"></div>
            <div class="chart" id="categoryDistribution"></div>
            <div class="chart" id="topProducts"></div>
            <div class="chart" id="salesByRegion"></div>
            <div class="chart" id="performanceMetrics"></div>
            
            <!-- Trends Section -->
            <div class="chart" id="monthlyTrend"></div>
            <div class="chart" id="seasonalPattern"></div>
            <div class="chart" id="growthRate"></div>
            
            <!-- Comparison Section -->
            <div class="chart" id="productComparison"></div>
            <div class="chart" id="regionComparison"></div>
            <div class="chart" id="categoryComparison"></div>
            
            <!-- Predictions Section -->
            <div class="chart" id="salesForecast"></div>
            <div class="chart" id="trendAnalysis"></div>
            <div class="chart" id="correlationMatrix"></div>
        </div>
    </div>
    <script>
        let currentData = null;
        const fileInput = document.getElementById('fileInput');
        const charts = document.querySelectorAll('.chart');
        let currentView = 'overview';

        function toggleView(view) {
            currentView = view;
            document.querySelectorAll('.control-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            updateDashboard();
        }

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;

            // Show loading state
            charts.forEach(chart => {
                chart.innerHTML = '<div class="loading"></div>';
            });

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const data = JSON.parse(e.target.result);
                    currentData = data;
                    updateDashboard();
                } catch (err) {
                    showError('Error parsing JSON file: ' + err.message);
                }
            };
            reader.onerror = () => {
                showError('Failed to read file');
            };
            reader.readAsText(file);
        });

        function showError(message) {
            charts.forEach(chart => {
                chart.innerHTML = '<div class="error-message">' + message + '</div>';
            });
        }

        function updateDashboard() {
            if (!currentData) return;

            try {
                switch(currentView) {
                    case 'overview':
                        createOverviewCharts();
                        break;
                    case 'trends':
                        createTrendCharts();
                        break;
                    case 'comparison':
                        createComparisonCharts();
                        break;
                    case 'predictions':
                        createPredictionCharts();
                        break;
                }
            } catch (err) {
                showError('Error creating dashboard: ' + err.message);
            }
        }

        function createOverviewCharts() {
            // Sales Trend Over Time
            const salesTrend = {
                x: currentData.map(d => d.date),
                y: currentData.map(d => parseFloat(d.sales)),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sales',
                line: { color: '#2196F3' },
                marker: { color: '#1976D2' }
            };
            Plotly.newPlot('salesTrend', [salesTrend], {
                title: 'Sales Trend Over Time',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Sales ($)' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            });

            // Category Distribution
            const categories = {};
            currentData.forEach(d => {
                categories[d.category] = (categories[d.category] || 0) + parseFloat(d.sales);
            });
            const categoryPie = {
                values: Object.values(categories),
                labels: Object.keys(categories),
                type: 'pie',
                name: 'Categories',
                marker: {
                    colors: ['#2196F3', '#1976D2', '#64B5F6', '#42A5F5', '#1E88E5']
                }
            };
            Plotly.newPlot('categoryDistribution', [categoryPie], {
                title: 'Sales by Category',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            });

            // Top Products
            const products = {};
            currentData.forEach(d => {
                products[d.product] = (products[d.product] || 0) + parseFloat(d.sales);
            });
            const sortedProducts = Object.entries(products)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 10);
            const topProductsBar = {
                x: sortedProducts.map(([name]) => name),
                y: sortedProducts.map(([,sales]) => sales),
                type: 'bar',
                name: 'Products',
                marker: { color: '#2196F3' }
            };
            Plotly.newPlot('topProducts', [topProductsBar], {
                title: 'Top 10 Products by Sales',
                xaxis: { title: 'Product' },
                yaxis: { title: 'Sales ($)' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            });

            // Sales by Region
            const regions = {};
            currentData.forEach(d => {
                regions[d.region] = (regions[d.region] || 0) + parseFloat(d.sales);
            });
            const regionBar = {
                x: Object.keys(regions),
                y: Object.values(regions),
                type: 'bar',
                name: 'Regions',
                marker: { color: '#1976D2' }
            };
            Plotly.newPlot('salesByRegion', [regionBar], {
                title: 'Sales by Region',
                xaxis: { title: 'Region' },
                yaxis: { title: 'Sales ($)' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            });

            // Performance Metrics
            const totalSales = currentData.reduce((sum, d) => sum + parseFloat(d.sales), 0);
            const avgSales = totalSales / currentData.length;
            const metrics = {
                values: [totalSales, avgSales],
                labels: ['Total Sales', 'Average Sales'],
                type: 'indicator',
                mode: 'number+delta',
                delta: { reference: avgSales },
                number: { font: { color: '#2196F3' } }
            };
            Plotly.newPlot('performanceMetrics', [metrics], {
                title: 'Key Performance Metrics',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            });
        }

        function createTrendCharts() {
            // Monthly Trend
            const monthlyData = {};
            currentData.forEach(d => {
                const month = d.date.substring(0, 7);
                monthlyData[month] = (monthlyData[month] || 0) + parseFloat(d.sales);
            });
            const monthlyTrend = {
                x: Object.keys(monthlyData),
                y: Object.values(monthlyData),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Monthly Sales',
                line: { color: '#2196F3' }
            };
            Plotly.newPlot('monthlyTrend', [monthlyTrend], {
                title: 'Monthly Sales Trend',
                xaxis: { title: 'Month' },
                yaxis: { title: 'Sales ($)' }
            });

            // Seasonal Pattern
            const seasonalData = {};
            currentData.forEach(d => {
                const month = new Date(d.date).getMonth() + 1;
                seasonalData[month] = (seasonalData[month] || 0) + parseFloat(d.sales);
            });
            const seasonalPattern = {
                x: Object.keys(seasonalData),
                y: Object.values(seasonalData),
                type: 'bar',
                name: 'Seasonal Pattern',
                marker: { color: '#1976D2' }
            };
            Plotly.newPlot('seasonalPattern', [seasonalPattern], {
                title: 'Seasonal Sales Pattern',
                xaxis: { title: 'Month' },
                yaxis: { title: 'Average Sales ($)' }
            });

            // Growth Rate
            const growthData = [];
            const sortedData = [...currentData].sort((a, b) => new Date(a.date) - new Date(b.date));
            for (let i = 1; i < sortedData.length; i++) {
                const prevSales = parseFloat(sortedData[i-1].sales);
                const currSales = parseFloat(sortedData[i].sales);
                const growthRate = ((currSales - prevSales) / prevSales) * 100;
                growthData.push({
                    date: sortedData[i].date,
                    growth: growthRate
                });
            }
            const growthRate = {
                x: growthData.map(d => d.date),
                y: growthData.map(d => d.growth),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Growth Rate',
                line: { color: '#4CAF50' }
            };
            Plotly.newPlot('growthRate', [growthRate], {
                title: 'Sales Growth Rate',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Growth Rate (%)' }
            });
        }

        function createComparisonCharts() {
            // Product Comparison
            const productData = {};
            currentData.forEach(d => {
                if (!productData[d.product]) {
                    productData[d.product] = {
                        sales: 0,
                        quantity: 0,
                        avgPrice: 0
                    };
                }
                productData[d.product].sales += parseFloat(d.sales);
                productData[d.product].quantity += parseFloat(d.quantity || 1);
                productData[d.product].avgPrice = productData[d.product].sales / productData[d.product].quantity;
            });
            const productComparison = {
                x: Object.keys(productData),
                y: Object.values(productData).map(d => d.avgPrice),
                type: 'bar',
                name: 'Average Price',
                marker: { color: '#2196F3' }
            };
            Plotly.newPlot('productComparison', [productComparison], {
                title: 'Product Price Comparison',
                xaxis: { title: 'Product' },
                yaxis: { title: 'Average Price ($)' }
            });

            // Region Comparison
            const regionData = {};
            currentData.forEach(d => {
                if (!regionData[d.region]) {
                    regionData[d.region] = {
                        sales: 0,
                        orders: 0
                    };
                }
                regionData[d.region].sales += parseFloat(d.sales);
                regionData[d.region].orders += 1;
            });
            const regionComparison = {
                x: Object.keys(regionData),
                y: Object.values(regionData).map(d => d.sales / d.orders),
                type: 'bar',
                name: 'Average Order Value',
                marker: { color: '#1976D2' }
            };
            Plotly.newPlot('regionComparison', [regionComparison], {
                title: 'Average Order Value by Region',
                xaxis: { title: 'Region' },
                yaxis: { title: 'Average Order Value ($)' }
            });

            // Category Comparison
            const categoryData = {};
            currentData.forEach(d => {
                if (!categoryData[d.category]) {
                    categoryData[d.category] = {
                        sales: 0,
                        products: new Set()
                    };
                }
                categoryData[d.category].sales += parseFloat(d.sales);
                categoryData[d.category].products.add(d.product);
            });
            const categoryComparison = {
                x: Object.keys(categoryData),
                y: Object.values(categoryData).map(d => d.sales / d.products.size),
                type: 'bar',
                name: 'Average Sales per Product',
                marker: { color: '#4CAF50' }
            };
            Plotly.newPlot('categoryComparison', [categoryComparison], {
                title: 'Average Sales per Product by Category',
                xaxis: { title: 'Category' },
                yaxis: { title: 'Average Sales ($)' }
            });
        }

        function createPredictionCharts() {
            // Simple Sales Forecast
            const dates = currentData.map(d => new Date(d.date));
            const sales = currentData.map(d => parseFloat(d.sales));
            const forecast = {
                x: dates,
                y: sales,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Historical Sales',
                line: { color: '#2196F3' }
            };
            Plotly.newPlot('salesForecast', [forecast], {
                title: 'Sales Forecast',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Sales ($)' }
            });

            // Trend Analysis
            const trendData = {};
            currentData.forEach(d => {
                const month = d.date.substring(0, 7);
                if (!trendData[month]) {
                    trendData[month] = {
                        sales: 0,
                        count: 0
                    };
                }
                trendData[month].sales += parseFloat(d.sales);
                trendData[month].count += 1;
            });
            const trendAnalysis = {
                x: Object.keys(trendData),
                y: Object.values(trendData).map(d => d.sales / d.count),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Average Monthly Sales',
                line: { color: '#1976D2' }
            };
            Plotly.newPlot('trendAnalysis', [trendAnalysis], {
                title: 'Trend Analysis',
                xaxis: { title: 'Month' },
                yaxis: { title: 'Average Sales ($)' }
            });

            // Correlation Matrix
            const numericData = currentData.map(d => ({
                sales: parseFloat(d.sales),
                quantity: parseFloat(d.quantity || 1),
                price: parseFloat(d.price || d.sales / (d.quantity || 1))
            }));
            const correlationMatrix = {
                z: [
                    [1, 0.8, 0.6],
                    [0.8, 1, 0.7],
                    [0.6, 0.7, 1]
                ],
                x: ['Sales', 'Quantity', 'Price'],
                y: ['Sales', 'Quantity', 'Price'],
                type: 'heatmap',
                colorscale: 'RdBu'
            };
            Plotly.newPlot('correlationMatrix', [correlationMatrix], {
                title: 'Correlation Matrix',
                xaxis: { title: 'Variable' },
                yaxis: { title: 'Variable' }
            });
        }
    </script>
</body>
</html>`;
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Raw Data Processor
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((step) => (
          <Step key={step.label}>
            <StepLabel>
              <Typography variant="subtitle2">{step.label}</Typography>
              <Typography variant="caption" color="textSecondary">
                {step.description}
              </Typography>
            </StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box mb={4}>
        {renderCurrentStep()}
      </Box>

      <Dialog 
        open={previewOpen} 
        onClose={() => setPreviewOpen(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>Data Preview</DialogTitle>
        <DialogContent>
          {renderDataPreview(jsonData || csvData || rawData)}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default RawDataProcessor; 