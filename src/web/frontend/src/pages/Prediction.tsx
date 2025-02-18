import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PredictionFormData {
  modelId: string;
  targetColumn: string;
  forecastHorizon?: number;
}

interface ModelMetadata {
  model_id: string;
  training_date: string;
  metrics: Record<string, number>;
}

const Prediction: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [formData, setFormData] = useState<PredictionFormData>({
    modelId: '',
    targetColumn: 'CV Gross Sales',
  });
  const [predictions, setPredictions] = useState<any>(null);

  // Fetch available models
  const { data: models, isLoading: isLoadingModels } = useQuery<ModelMetadata[]>(
    'models',
    async () => {
      const response = await axios.get('/api/models');
      return response.data;
    }
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  const predictMutation = useMutation(
    async () => {
      if (!file) return;

      const formDataToSend = new FormData();
      formDataToSend.append('file', file);
      formDataToSend.append(
        'request',
        JSON.stringify({
          model_id: formData.modelId,
          target_column: formData.targetColumn,
          forecast_horizon: formData.forecastHorizon,
        })
      );

      const response = await axios.post('/api/predict', formDataToSend, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    },
    {
      onSuccess: (data) => {
        setPredictions(data);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    predictMutation.mutate();
  };

  const renderChart = () => {
    if (!predictions) return null;

    const historicalData = predictions.historical_predictions;
    const futureData = predictions.future_predictions;

    const data = {
      labels: [
        ...historicalData.map((d: any) => d.date),
        ...futureData.map((d: any) => d.date),
      ],
      datasets: [
        {
          label: 'Actual',
          data: historicalData.map((d: any) => d.actual),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
        },
        {
          label: 'Historical Predictions',
          data: historicalData.map((d: any) => d.prediction),
          borderColor: 'rgb(255, 99, 132)',
          tension: 0.1,
        },
        {
          label: 'Future Predictions',
          data: [
            ...Array(historicalData.length).fill(null),
            ...futureData.map((d: any) => d.prediction),
          ],
          borderColor: 'rgb(54, 162, 235)',
          tension: 0.1,
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Sales Forecast',
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    return <Line data={data} options={options} />;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Generate Predictions
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed #ccc',
              borderRadius: 2,
              p: 3,
              textAlign: 'center',
              cursor: 'pointer',
              mb: 3,
              '&:hover': {
                borderColor: 'primary.main',
              },
            }}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <Typography>Drop the CSV file here</Typography>
            ) : (
              <Typography>
                Drag and drop a CSV file here, or click to select
              </Typography>
            )}
            {file && (
              <Typography variant="body2" color="primary" sx={{ mt: 1 }}>
                Selected file: {file.name}
              </Typography>
            )}
          </Box>

          <form onSubmit={handleSubmit}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Model</InputLabel>
              <Select
                value={formData.modelId}
                label="Model"
                onChange={(e) =>
                  setFormData({ ...formData, modelId: e.target.value })
                }
                required
              >
                {isLoadingModels ? (
                  <MenuItem disabled>Loading models...</MenuItem>
                ) : (
                  models?.map((model) => (
                    <MenuItem key={model.model_id} value={model.model_id}>
                      {`Model ${model.model_id.slice(0, 8)} (${
                        new Date(model.training_date).toLocaleDateString()
                      })`}
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <TextField
                label="Target Column"
                value={formData.targetColumn}
                onChange={(e) =>
                  setFormData({ ...formData, targetColumn: e.target.value })
                }
                required
              />
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <TextField
                label="Forecast Horizon (Optional)"
                type="number"
                value={formData.forecastHorizon || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    forecastHorizon: parseInt(e.target.value),
                  })
                }
              />
            </FormControl>

            <Button
              variant="contained"
              color="primary"
              type="submit"
              disabled={!file || !formData.modelId || predictMutation.isLoading}
              fullWidth
            >
              {predictMutation.isLoading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Generate Predictions'
              )}
            </Button>
          </form>

          {predictMutation.isError && (
            <Typography color="error" sx={{ mt: 2 }}>
              Error: {(predictMutation.error as Error).message}
            </Typography>
          )}
        </CardContent>
      </Card>

      {predictions && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Forecast Results
            </Typography>
            {renderChart()}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Prediction;
