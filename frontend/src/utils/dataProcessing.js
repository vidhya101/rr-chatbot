export const processData = (data, params) => {
  let processed = [...data];

  // Remove null values if specified
  if (params.removeNulls) {
    processed = processed.filter(row => 
      Object.values(row).every(val => val !== null && val !== undefined)
    );
  }

  // Apply date formatting
  if (params.dateColumn) {
    processed = processed.map(row => ({
      ...row,
      [params.dateColumn]: formatDate(row[params.dateColumn], params.dateFormat)
    }));
  }

  // Apply aggregation
  if (params.aggregation !== 'none') {
    processed = aggregateData(processed, params.aggregation, params.groupBy);
  }

  // Apply sorting
  if (params.sortBy !== 'none') {
    processed = sortData(processed, params.sortBy, params.sortOrder);
  }

  return processed;
};

export const aggregateData = (data, aggFunction, groupBy) => {
  // Implementation for different aggregation functions
  const aggregations = {
    count: values => values.length,
    sum: values => values.reduce((a, b) => a + b, 0),
    average: values => values.reduce((a, b) => a + b, 0) / values.length,
    median: values => {
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    },
    stddev: values => {
      const avg = aggregations.average(values);
      const squareDiffs = values.map(value => Math.pow(value - avg, 2));
      return Math.sqrt(aggregations.average(squareDiffs));
    }
    // ... other aggregation functions ...
  };
  
  return groupBy ? 
    groupAndAggregate(data, groupBy, aggregations[aggFunction]) :
    aggregations[aggFunction](data);
}; 