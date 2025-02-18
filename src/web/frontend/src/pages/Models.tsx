import React from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';
import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  Typography,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';

interface ModelMetadata {
  model_id: string;
  training_date: string;
  metrics: Record<string, number>;
  parameters: Record<string, any>;
}

const Models: React.FC = () => {
  const { data: models, isLoading } = useQuery<ModelMetadata[]>(
    'models',
    async () => {
      const response = await axios.get('/api/models');
      return response.data;
    }
  );

  const columns: GridColDef[] = [
    {
      field: 'model_id',
      headerName: 'Model ID',
      width: 200,
      valueGetter: (params) => params.row.model_id.slice(0, 8),
    },
    {
      field: 'training_date',
      headerName: 'Training Date',
      width: 200,
      valueGetter: (params) =>
        new Date(params.row.training_date).toLocaleString(),
    },
    {
      field: 'target_column',
      headerName: 'Target Column',
      width: 150,
      valueGetter: (params) => params.row.parameters.target_column,
    },
    {
      field: 'meta_model_type',
      headerName: 'Meta Model',
      width: 150,
      valueGetter: (params) => params.row.parameters.meta_model_type,
    },
    {
      field: 'forecast_horizon',
      headerName: 'Horizon',
      width: 100,
      valueGetter: (params) => params.row.parameters.forecast_horizon,
    },
    {
      field: 'train_rmse',
      headerName: 'RMSE',
      width: 120,
      valueGetter: (params) =>
        params.row.metrics.train_rmse?.toFixed(4) || 'N/A',
    },
    {
      field: 'train_mae',
      headerName: 'MAE',
      width: 120,
      valueGetter: (params) => params.row.metrics.train_mae?.toFixed(4) || 'N/A',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trained Models
      </Typography>

      <Card>
        <CardContent>
          {isLoading ? (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: 400,
              }}
            >
              <CircularProgress />
            </Box>
          ) : (
            <DataGrid
              rows={models || []}
              columns={columns}
              getRowId={(row) => row.model_id}
              autoHeight
              initialState={{
                pagination: {
                  paginationModel: {
                    pageSize: 10,
                  },
                },
                sorting: {
                  sortModel: [{ field: 'training_date', sort: 'desc' }],
                },
              }}
              pageSizeOptions={[5, 10, 25]}
            />
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default Models;
