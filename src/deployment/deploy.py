"""
Script for deploying sales forecasting models.
"""
import argparse
import logging
from .manager import ModelDeploymentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description='Deploy sales forecasting model')
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model to deploy'
    )
    
    parser.add_argument(
        '--deployment-name',
        type=str,
        help='Optional name for deployment'
    )
    
    parser.add_argument(
        '--deployment-type',
        type=str,
        choices=['AzureML', 'Local'],
        default='AzureML',
        help='Type of deployment'
    )
    
    parser.add_argument(
        '--environment',
        type=str,
        choices=['development', 'staging', 'production'],
        default='development',
        help='Deployment environment'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['deploy', 'delete', 'list'],
        default='deploy',
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize deployment manager
        manager = ModelDeploymentManager(env=args.environment)
        
        if args.action == 'deploy':
            # Deploy model
            endpoint_url = manager.deploy_model(
                model_name=args.model_name,
                deployment_name=args.deployment_name,
                deployment_type=args.deployment_type
            )
            
            logger.info(f"Model deployed successfully")
            logger.info(f"Endpoint URL: {endpoint_url}")
            
        elif args.action == 'delete':
            if not args.deployment_name:
                raise ValueError("Deployment name is required for delete action")
                
            # Delete deployment
            manager.delete_deployment(
                deployment_name=args.deployment_name,
                deployment_type=args.deployment_type
            )
            
            logger.info(f"Deployment {args.deployment_name} deleted successfully")
            
        elif args.action == 'list':
            # List deployments
            deployments = manager.list_deployments()
            
            logger.info("\nActive Deployments:")
            for name, info in deployments.items():
                logger.info(f"\nDeployment: {name}")
                logger.info(f"Type: {info['type']}")
                logger.info(f"State: {info['state']}")
                logger.info(f"Endpoint: {info['endpoint']}")
                logger.info(f"Created: {info['created_time']}")
                
                if 'compute' in info:
                    logger.info(f"CPU Cores: {info['compute']['cpu_cores']}")
                    logger.info(f"Memory (GB): {info['compute']['memory_gb']}")
        
    except Exception as e:
        logger.error(f"Error in deployment: {str(e)}")
        raise

if __name__ == '__main__':
    main()
