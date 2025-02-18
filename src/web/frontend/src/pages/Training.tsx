import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from 'react-query';
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

interface TrainingFormData {
  targetColumn: string;
  forecastHorizon: number;
  metaModelType: string;
}

const Training: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [formData, setFormData] = useState<TrainingFormData>({
    targetColumn: 'CV Gross Sales',
    forecastHorizon: 13,
    metaModelType: 'xgboost',
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  const trainMutation = useMutation(
    async () => {
      if (!file) return;

      const formDataToSend = new FormData();
      formDataToSend.append('file', file);
      formDataToSend.append(
        'request',
        JSON.stringify({
          target_column: formData.targetColumn,
          forecast_horizon: formData.forecastHorizon,
          meta_model_type: formData.metaModelType,
        })
      );

      const response = await axios.post('/api/train', formDataToSend, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    },
    {
      onSuccess: () => {
        navigate('/models');
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    trainMutation.mutate();
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Train New Model
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
              <TextField
                label="Target Column"
                value={formData.targetColumn}
                onChange={(e) =>
                  setFormData({ ...formData, targetColumn: e.target.value })
                }
                required
              />
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <TextField
                label="Forecast Horizon"
                type="number"
                value={formData.forecastHorizon}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    forecastHorizon: parseInt(e.target.value),
                  })
                }
                required
              />
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Meta Model Type</InputLabel>
              <Select
                value={formData.metaModelType}
                label="Meta Model Type"
                onChange={(e) =>
                  setFormData({ ...formData, metaModelType: e.target.value })
                }
                required
              >
                <MenuItem value="xgboost">XGBoost</MenuItem>
                <MenuItem value="rf">Random Forest</MenuItem>
                <MenuItem value="gbm">Gradient Boosting</MenuItem>
              </Select>
            </FormControl>

            <Button
              variant="contained"
              color="primary"
              type="submit"
              disabled={!file || trainMutation.isLoading}
              fullWidth
            >
              {trainMutation.isLoading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Train Model'
              )}
            </Button>
          </form>

          {trainMutation.isError && (
            <Typography color="error" sx={{ mt: 2 }}>
              Error: {(trainMutation.error as Error).message}
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default Training;
