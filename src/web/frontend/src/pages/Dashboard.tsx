import React from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';

interface ModelMetadata {
  model_id: string;
  training_date: string;
  metrics: Record<string, number>;
}

const Dashboard: React.FC = () => {
  const { data: models, isLoading } = useQuery<ModelMetadata[]>(
    'models',
    async () => {
      const response = await axios.get('/api/models');
      return response.data;
    }
  );

  const stats = {
    totalModels: models?.length || 0,
    bestRMSE: models
      ? Math.min(...models.map((m) => m.metrics.train_rmse || Infinity))
      : 0,
    latestModel: models
      ? new Date(
          Math.max(...models.map((m) => new Date(m.training_date).getTime()))
        ).toLocaleDateString()
      : 'N/A',
    averageMAE: models
      ? (
          models.reduce((acc, m) => acc + (m.metrics.train_mae || 0), 0) /
          models.length
        ).toFixed(4)
      : 0,
  };

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
  }> = ({ title, value, icon, color }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box>
            <Typography variant="h6" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4">{value}</Typography>
          </Box>
          <Box
            sx={{
              backgroundColor: color,
              borderRadius: '50%',
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (isLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Models"
            value={stats.totalModels}
            icon={<StorageIcon sx={{ color: 'white' }} />}
            color="#1976d2"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Best RMSE"
            value={stats.bestRMSE.toFixed(4)}
            icon={<SpeedIcon sx={{ color: 'white' }} />}
            color="#2e7d32"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Latest Model"
            value={stats.latestModel}
            icon={<TimelineIcon sx={{ color: 'white' }} />}
            color="#ed6c02"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Average MAE"
            value={stats.averageMAE}
            icon={<AssessmentIcon sx={{ color: 'white' }} />}
            color="#9c27b0"
          />
        </Grid>
      </Grid>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Model Performance Overview
          </Typography>
          <Typography variant="body1">
            The sales forecasting system currently has {stats.totalModels} trained
            models. The best performing model achieved an RMSE of{' '}
            {stats.bestRMSE.toFixed(4)}, while the average MAE across all models
            is {stats.averageMAE}. The most recent model was trained on{' '}
            {stats.latestModel}.
          </Typography>
        </CardContent>
      </Card>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Quick Actions
          </Typography>
          <Typography variant="body1">
            Use the navigation menu to:
            <ul>
              <li>Train new models with custom parameters</li>
              <li>Generate predictions using trained models</li>
              <li>View detailed model metrics and comparisons</li>
            </ul>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Dashboard;
