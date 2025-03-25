import React, { useState, useEffect } from 'react';
import {
  Grid,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Box,
  LinearProgress,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  SpeedDial,
  SpeedDialIcon,
  SpeedDialAction,
  Slider
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Timeline as TimelineIcon,
  BugReport as BugReportIcon,
  NetworkCheck as NetworkIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  Lightbulb as LightbulbIcon,
  Assessment as AssessmentIcon,
  Event as EventIcon,
  PlayArrow as PlayArrowIcon
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend
);

const MonitoringDashboard = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [detailedMetrics, setDetailedMetrics] = useState(null);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [historicalData, setHistoricalData] = useState(null);
  const [thresholds, setThresholds] = useState(null);
  const [thresholdDialogOpen, setThresholdDialogOpen] = useState(false);
  const [selectedThreshold, setSelectedThreshold] = useState(null);
  const [timeRange, setTimeRange] = useState('1h');
  const [advancedMetrics, setAdvancedMetrics] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [expandedAccordion, setExpandedAccordion] = useState('panel1');
  const [systemEvents, setSystemEvents] = useState([]);
  const [performanceTrends, setPerformanceTrends] = useState(null);
  const [availableActions, setAvailableActions] = useState([]);
  const [actionDialogOpen, setActionDialogOpen] = useState(false);
  const [selectedAction, setSelectedAction] = useState(null);
  const [speedDialOpen, setSpeedDialOpen] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [
        healthResponse,
        metricsResponse,
        performanceResponse,
        alertsResponse,
        historicalResponse,
        thresholdsResponse,
        advancedResponse,
        predictionsResponse,
        recommendationsResponse,
        eventsResponse,
        trendsResponse,
        actionsResponse
      ] = await Promise.all([
        fetch('/api/system/health'),
        fetch('/api/system/metrics/detailed'),
        fetch('/api/system/performance'),
        fetch('/api/system/alerts'),
        fetch('/api/system/metrics/historical'),
        fetch('/api/system/alerts/thresholds'),
        fetch('/api/system/metrics/advanced'),
        fetch('/api/system/predictions'),
        fetch('/api/system/recommendations'),
        fetch('/api/system/events'),
        fetch('/api/system/performance/trends'),
        fetch('/api/system/actions')
      ]);

      if (!healthResponse.ok || !metricsResponse.ok || !performanceResponse.ok || 
          !alertsResponse.ok || !historicalResponse.ok || !thresholdsResponse.ok ||
          !advancedResponse.ok || !predictionsResponse.ok || !recommendationsResponse.ok ||
          !eventsResponse.ok || !trendsResponse.ok || !actionsResponse.ok) {
        throw new Error('Failed to fetch monitoring data');
      }

      const healthData = await healthResponse.json();
      const metricsData = await metricsResponse.json();
      const performanceData = await performanceResponse.json();
      const alertsData = await alertsResponse.json();
      const historicalData = await historicalResponse.json();
      const thresholdsData = await thresholdsResponse.json();
      const advancedData = await advancedResponse.json();
      const predictionsData = await predictionsResponse.json();
      const recommendationsData = await recommendationsResponse.json();
      const eventsData = await eventsResponse.json();
      const trendsData = await trendsResponse.json();
      const actionsData = await actionsResponse.json();

      setSystemHealth(healthData);
      setDetailedMetrics(metricsData);
      setPerformanceMetrics(performanceData);
      setAlerts(alertsData.alerts);
      setHistoricalData(historicalData.historical_data);
      setThresholds(thresholdsData.thresholds);
      setAdvancedMetrics(advancedData);
      setPredictions(predictionsData.predictions);
      setRecommendations(recommendationsData.recommendations);
      setSystemEvents(eventsData.events);
      setPerformanceTrends(trendsData.trends);
      setAvailableActions(actionsData.available_actions);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching monitoring data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'ok':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'info';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ok':
        return <CheckCircleIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <ErrorIcon color="info" />;
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleThresholdChange = async (component, newValue) => {
    try {
      const response = await fetch('/api/system/alerts/thresholds', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          [component]: newValue
        })
      });

      if (!response.ok) {
        throw new Error('Failed to update threshold');
      }

      const data = await response.json();
      setThresholds(data.thresholds);
      setThresholdDialogOpen(false);
    } catch (err) {
      console.error('Error updating threshold:', err);
    }
  };

  const renderHistoricalChart = (data, label, color) => {
    if (!data) return null;

    const timestamps = Object.keys(data).sort();
    const values = timestamps.map(t => data[t]);

    return {
      labels: timestamps.map(t => new Date(t).toLocaleTimeString()),
      datasets: [
        {
          label,
          data: values,
          borderColor: color,
          backgroundColor: color + '20',
          fill: true,
          tension: 0.4
        }
      ]
    };
  };

  const renderThresholdDialog = () => (
    <Dialog open={thresholdDialogOpen} onClose={() => setThresholdDialogOpen(false)}>
      <DialogTitle>Update Alert Threshold</DialogTitle>
      <DialogContent>
        <Box sx={{ mt: 2 }}>
          <Typography gutterBottom>
            Current threshold: {thresholds[selectedThreshold]}
          </Typography>
          <Slider
            value={thresholds[selectedThreshold]}
            onChange={(_, value) => handleThresholdChange(selectedThreshold, value)}
            min={0}
            max={100}
            step={1}
            marks
            valueLabelDisplay="auto"
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setThresholdDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpandedAccordion(isExpanded ? panel : false);
  };

  const renderPredictionsCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <TrendingUpIcon color="primary" />
          <Typography variant="h6" ml={1}>
            Resource Usage Predictions
          </Typography>
        </Box>
        {predictions && Object.entries(predictions).map(([resource, data]) => (
          <Box key={resource} mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              {resource.toUpperCase()}
            </Typography>
            <Box display="flex" alignItems="center">
              <Typography variant="body2" mr={2}>
                Next Hour: {data.next_hour?.toFixed(1)}%
              </Typography>
              <Chip
                label={data.trend}
                color={data.trend === 'increasing' ? 'error' : 'success'}
                size="small"
              />
            </Box>
          </Box>
        ))}
      </CardContent>
    </Card>
  );

  const renderRecommendationsCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <LightbulbIcon color="primary" />
          <Typography variant="h6" ml={1}>
            System Recommendations
          </Typography>
        </Box>
        {recommendations.length === 0 ? (
          <Typography variant="body2" color="textSecondary">
            No recommendations at this time
          </Typography>
        ) : (
          <List>
            {recommendations.map((rec, index) => (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemIcon>
                    {rec.type === 'cpu' && <SpeedIcon color={rec.severity === 'high' ? 'error' : 'warning'} />}
                    {rec.type === 'memory' && <MemoryIcon color={rec.severity === 'high' ? 'error' : 'warning'} />}
                    {rec.type === 'disk' && <StorageIcon color={rec.severity === 'high' ? 'error' : 'warning'} />}
                    {rec.type === 'network' && <NetworkIcon color={rec.severity === 'high' ? 'error' : 'warning'} />}
                  </ListItemIcon>
                  <ListItemText
                    primary={rec.message}
                    secondary={rec.action}
                  />
                </ListItem>
                {index < recommendations.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );

  const renderAdvancedMetricsCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <AssessmentIcon color="primary" />
          <Typography variant="h6" ml={1}>
            Advanced Metrics
          </Typography>
        </Box>
        {advancedMetrics && (
          <Accordion expanded={expandedAccordion === 'panel1'} onChange={handleAccordionChange('panel1')}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Process Metrics</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2">Threads: {advancedMetrics.process.threads}</Typography>
                  <Typography variant="body2">File Descriptors: {advancedMetrics.process.file_descriptors}</Typography>
                  <Typography variant="body2">Priority: {advancedMetrics.process.priority}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">Status: {advancedMetrics.process.status}</Typography>
                  <Typography variant="body2">Username: {advancedMetrics.process.username}</Typography>
                  <Typography variant="body2">CPU Affinity: {advancedMetrics.process.cpu_affinity.join(', ')}</Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}
      </CardContent>
    </Card>
  );

  const handleAction = async (action) => {
    try {
      const response = await fetch('/api/system/actions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: action.name })
      });

      if (!response.ok) {
        throw new Error('Failed to perform action');
      }

      const result = await response.json();
      // Show success message
      setActionDialogOpen(false);
      fetchData(); // Refresh data
    } catch (err) {
      console.error('Error performing action:', err);
    }
  };

  const renderEventsCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <EventIcon color="primary" />
          <Typography variant="h6" ml={1}>
            System Events
          </Typography>
        </Box>
        <List>
          {systemEvents.map((event, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  {event.type === 'critical' && <ErrorIcon color="error" />}
                  {event.type === 'log' && <BugReportIcon color="warning" />}
                </ListItemIcon>
                <ListItemText
                  primary={event.message}
                  secondary={`${event.category} - ${new Date(event.timestamp).toLocaleString()}`}
                />
              </ListItem>
              {event.details && (
                <Box ml={4} mb={2}>
                  {Object.entries(event.details).map(([key, value]) => (
                    <Typography key={key} variant="body2" color="textSecondary">
                      {key}: {value}
                    </Typography>
                  ))}
                </Box>
              )}
              {index < systemEvents.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      </CardContent>
    </Card>
  );

  const renderTrendsCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <TrendingUpIcon color="primary" />
          <Typography variant="h6" ml={1}>
            Performance Trends
          </Typography>
        </Box>
        {performanceTrends && Object.entries(performanceTrends).map(([metric, trend]) => (
          trend && (
            <Box key={metric} mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                {metric.toUpperCase()}
              </Typography>
              <Box display="flex" alignItems="center">
                <Typography variant="body2" mr={2}>
                  Direction: {trend.direction}
                </Typography>
                <Typography variant="body2" mr={2}>
                  Volatility: {trend.volatility.toFixed(2)}
                </Typography>
                <Chip
                  label={trend.direction}
                  color={trend.direction === 'increasing' ? 'error' : 'success'}
                  size="small"
                />
              </Box>
            </Box>
          )
        ))}
      </CardContent>
    </Card>
  );

  const renderActionDialog = () => (
    <Dialog open={actionDialogOpen} onClose={() => setActionDialogOpen(false)}>
      <DialogTitle>Confirm Action</DialogTitle>
      <DialogContent>
        <Typography>
          Are you sure you want to perform the following action?
        </Typography>
        <Typography variant="subtitle1" color="primary" sx={{ mt: 2 }}>
          {selectedAction?.description}
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setActionDialogOpen(false)}>Cancel</Button>
        <Button onClick={() => handleAction(selectedAction)} color="primary">
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );

  const renderSpeedDial = () => (
    <SpeedDial
      ariaLabel="System Actions"
      sx={{ position: 'fixed', bottom: 16, right: 16 }}
      icon={<SpeedDialIcon />}
      onClose={() => setSpeedDialOpen(false)}
      onOpen={() => setSpeedDialOpen(true)}
      open={speedDialOpen}
    >
      {availableActions.map((action) => (
        <SpeedDialAction
          key={action.name}
          icon={<PlayArrowIcon />}
          tooltipTitle={action.description}
          onClick={() => {
            setSelectedAction(action);
            setActionDialogOpen(true);
            setSpeedDialOpen(false);
          }}
        />
      ))}
    </SpeedDial>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">System Monitoring</Typography>
        <Box>
          <Tooltip title="Update thresholds">
            <IconButton onClick={() => setThresholdDialogOpen(true)}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh data">
            <IconButton onClick={fetchData}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Box display="flex" alignItems="center" mb={3}>
        <FormControl sx={{ minWidth: 120, mr: 2 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            label="Time Range"
          >
            <MenuItem value="1h">Last Hour</MenuItem>
            <MenuItem value="6h">Last 6 Hours</MenuItem>
            <MenuItem value="24h">Last 24 Hours</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab icon={<CheckCircleIcon />} label="Health" />
        <Tab icon={<TimelineIcon />} label="Performance" />
        <Tab icon={<NetworkIcon />} label="Network" />
        <Tab icon={<BugReportIcon />} label="Alerts" />
      </Tabs>

      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* System Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Status
                </Typography>
                <Box display="flex" alignItems="center" mb={2}>
                  {getStatusIcon(systemHealth?.status)}
                  <Typography variant="body1" ml={1}>
                    Status: {systemHealth?.status}
                  </Typography>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Last Updated: {lastUpdated?.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Database Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Database Status
                </Typography>
                <Box display="flex" alignItems="center" mb={2}>
                  {getStatusIcon(systemHealth?.database)}
                  <Typography variant="body1" ml={1}>
                    Status: {systemHealth?.database}
                  </Typography>
                </Box>
                <Typography variant="body2">
                  Active Connections: {systemHealth?.system?.active_connections}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Predictions Card */}
          <Grid item xs={12} md={6}>
            {renderPredictionsCard()}
          </Grid>

          {/* Recommendations Card */}
          <Grid item xs={12} md={6}>
            {renderRecommendationsCard()}
          </Grid>

          {/* Advanced Metrics Card */}
          <Grid item xs={12}>
            {renderAdvancedMetricsCard()}
          </Grid>

          {/* Events Card */}
          <Grid item xs={12} md={6}>
            {renderEventsCard()}
          </Grid>

          {/* Trends Card */}
          <Grid item xs={12} md={6}>
            {renderTrendsCard()}
          </Grid>

          {/* Historical Usage Charts */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Historical Resource Usage
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Box height={300}>
                      <Line
                        data={renderHistoricalChart(historicalData?.cpu, 'CPU Usage', '#2196f3')}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            y: {
                              beginAtZero: true,
                              max: 100
                            }
                          }
                        }}
                      />
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box height={300}>
                      <Line
                        data={renderHistoricalChart(historicalData?.memory, 'Memory Usage', '#4caf50')}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            y: {
                              beginAtZero: true,
                              max: 100
                            }
                          }
                        }}
                      />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* System Resources */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <MemoryIcon color="primary" />
                  <Typography variant="h6" ml={1}>
                    CPU Usage
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.system?.cpu_percent}
                  color={systemHealth?.system?.cpu_percent > 80 ? 'error' : 'primary'}
                />
                <Typography variant="body2" mt={1}>
                  {systemHealth?.system?.cpu_percent}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <StorageIcon color="primary" />
                  <Typography variant="h6" ml={1}>
                    Memory Usage
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.system?.memory_percent}
                  color={systemHealth?.system?.memory_percent > 80 ? 'error' : 'primary'}
                />
                <Typography variant="body2" mt={1}>
                  {systemHealth?.system?.memory_percent}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <SpeedIcon color="primary" />
                  <Typography variant="h6" ml={1}>
                    Disk Usage
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.system?.disk_percent}
                  color={systemHealth?.system?.disk_percent > 80 ? 'error' : 'primary'}
                />
                <Typography variant="body2" mt={1}>
                  {systemHealth?.system?.disk_percent}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && performanceMetrics && (
        <Grid container spacing={3}>
          {/* Slow Queries */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Slow Queries
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Query</TableCell>
                        <TableCell>Calls</TableCell>
                        <TableCell>Total Time</TableCell>
                        <TableCell>Mean Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {performanceMetrics.slow_queries.map((query, index) => (
                        <TableRow key={index}>
                          <TableCell>{query.query}</TableCell>
                          <TableCell>{query.calls}</TableCell>
                          <TableCell>{query.total_time.toFixed(2)}s</TableCell>
                          <TableCell>{query.mean_time.toFixed(2)}s</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Endpoint Performance */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Endpoint Performance
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Endpoint</TableCell>
                        <TableCell>Method</TableCell>
                        <TableCell>Average Duration</TableCell>
                        <TableCell>Total Requests</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {performanceMetrics.endpoint_performance.endpoints.map((endpoint, index) => (
                        <TableRow key={index}>
                          <TableCell>{endpoint.path}</TableCell>
                          <TableCell>{endpoint.method}</TableCell>
                          <TableCell>{endpoint.avg_duration.toFixed(2)}s</TableCell>
                          <TableCell>{endpoint.requests}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && detailedMetrics && (
        <Grid container spacing={3}>
          {/* Network I/O */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Network I/O
                </Typography>
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography variant="body2">
                    Bytes Sent: {(detailedMetrics.network.bytes_sent / 1024 / 1024).toFixed(2)} MB
                  </Typography>
                  <Typography variant="body2">
                    Bytes Received: {(detailedMetrics.network.bytes_recv / 1024 / 1024).toFixed(2)} MB
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">
                    Packets Sent: {detailedMetrics.network.packets_sent}
                  </Typography>
                  <Typography variant="body2">
                    Packets Received: {detailedMetrics.network.packets_recv}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Network Errors */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Network Errors
                </Typography>
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography variant="body2" color="error">
                    Errors In: {detailedMetrics.network.errin}
                  </Typography>
                  <Typography variant="body2" color="error">
                    Errors Out: {detailedMetrics.network.errout}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="warning">
                    Drops In: {detailedMetrics.network.dropin}
                  </Typography>
                  <Typography variant="body2" color="warning">
                    Drops Out: {detailedMetrics.network.dropout}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 3 && (
        <Grid container spacing={3}>
          {/* System Alerts */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Alerts
                </Typography>
                {alerts.length === 0 ? (
                  <Typography variant="body2" color="textSecondary">
                    No active alerts
                  </Typography>
                ) : (
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Type</TableCell>
                          <TableCell>Component</TableCell>
                          <TableCell>Message</TableCell>
                          <TableCell>Timestamp</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {alerts.map((alert, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Chip
                                label={alert.type}
                                color={alert.type === 'error' ? 'error' : 'warning'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{alert.component}</TableCell>
                            <TableCell>{alert.message}</TableCell>
                            <TableCell>
                              {new Date(alert.timestamp).toLocaleString()}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {renderThresholdDialog()}
      {renderActionDialog()}
      {renderSpeedDial()}
    </Box>
  );
};

export default MonitoringDashboard; 