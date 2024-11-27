import asyncio
import datetime
import logging
from typing import *

# AWS
try:
    import boto3
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

# Azure
try:
    from azure.mgmt.monitor import MonitorManagementClient
    from azure.identity import ClientSecretCredential
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

# Google Cloud
try:
    from google.cloud import monitoring_v3
    from google.oauth2 import service_account
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

# DigitalOcean
try:
    import digitalocean
    HAS_DO = True
except ImportError:
    HAS_DO = False

class CloudCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.collectors = {}
        
        # Only register available collectors
        if HAS_AWS:
            self.collectors['aws'] = self.collect_aws_logs
        if HAS_AZURE:
            self.collectors['azure'] = self.collect_azure_logs
        if HAS_GCP:
            self.collectors['gcp'] = self.collect_gcp_logs
        if HAS_DO:
            self.collectors['digitalocean'] = self.collect_digitalocean_logs
            
        if not self.collectors:
            logging.warning("No cloud collectors available. Please install required packages.")

    async def start_collection(self):
        tasks = []
        for cloud_provider in self.config['enabled_providers']:
            if cloud_provider in self.collectors:
                tasks.append(self.collectors[cloud_provider]())
        
        await asyncio.gather(*tasks)

    async def collect_aws_logs(self):
        try:
            cloudwatch = boto3.client('cloudwatch',
                aws_access_key_id=self.config['aws']['access_key'],
                aws_secret_access_key=self.config['aws']['secret_key'],
                region_name=self.config['aws']['region']
            )

            while True:
                response = cloudwatch.get_metric_data(
                    MetricDataQueries=[
                        {
                            'Id': 'm1',
                            'MetricStat': {
                                'Metric': {
                                    'Namespace': 'AWS/EC2',
                                    'MetricName': 'CPUUtilization'
                                },
                                'Period': 300,
                                'Stat': 'Average'
                            }
                        }
                    ],
                    StartTime='2024-03-20T00:00:00Z',
                    EndTime='2024-03-20T01:00:00Z'
                )
                
                for result in response['MetricDataResults']:
                    yield {
                        'timestamp': datetime.utcnow(),
                        'source': 'aws',
                        'event_type': 'metric',
                        'message': str(result),
                        'raw_data': result
                    }
                
                await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logging.error(f"Error collecting AWS logs: {e}")

    async def collect_azure_logs(self):
        try:
            # Initialize Azure Monitor client
            credential = ClientSecretCredential(
                tenant_id=self.config['azure']['tenant_id'],
                client_id=self.config['azure']['client_id'],
                client_secret=self.config['azure']['client_secret']
            )
            monitor_client = MonitorManagementClient(credential, self.config['azure']['subscription_id'])

            while True:
                # Get metrics for all VMs in subscription
                metrics = monitor_client.metrics.list(
                    resource_uri="/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Compute/virtualMachines".format(
                        self.config['azure']['subscription_id'],
                        self.config['azure']['resource_group']
                    ),
                    timespan="{}/{}".format(
                        (datetime.utcnow() - datetime.timedelta(hours=1)).isoformat(),
                        datetime.utcnow().isoformat()
                    ),
                    interval='PT5M',
                    metricnames='Percentage CPU',
                    aggregation='Average'
                )

                for metric in metrics.value:
                    for timeseries in metric.timeseries:
                        for data in timeseries.data:
                            yield {
                                'timestamp': datetime.utcnow(),
                                'source': 'azure',
                                'event_type': 'metric',
                                'message': f"CPU Usage: {data.average}%",
                                'raw_data': {
                                    'metric_name': metric.name.value,
                                    'average': data.average,
                                    'timestamp': data.time_stamp
                                }
                            }

                await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logging.error(f"Error collecting Azure logs: {e}")
        pass

    async def collect_gcp_logs(self):
        try:
            # Initialize GCP monitoring client
            credentials = service_account.Credentials.from_service_account_file(
                self.config['gcp']['credentials_path']
            )
            client = monitoring_v3.MetricServiceClient(credentials=credentials)
            project_name = f"projects/{self.config['gcp']['project_id']}"

            while True:
                # Create time interval for the last hour
                now = datetime.utcnow()
                seconds = int(now.timestamp())
                interval = monitoring_v3.TimeInterval({
                    'end_time': {'seconds': seconds},
                    'start_time': {'seconds': seconds - 3600},
                })

                # Query CPU usage metrics
                results = client.list_time_series(
                    request={
                        'name': project_name,
                        'filter': 'metric.type = "compute.googleapis.com/instance/cpu/utilization"',
                        'interval': interval,
                        'view': monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                    }
                )

                for result in results:
                    for point in result.points:
                        yield {
                            'timestamp': datetime.utcnow(),
                            'source': 'gcp',
                            'event_type': 'metric',
                            'message': f"CPU Usage: {point.value.double_value * 100}%",
                            'raw_data': {
                                'metric_name': result.metric.type,
                                'value': point.value.double_value * 100,
                                'timestamp': point.interval.end_time.seconds
                            }
                        }

                await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logging.error(f"Error collecting GCP logs: {e}")
        pass

    async def collect_digitalocean_logs(self):
        try:
            # Initialize DigitalOcean client
            client = digitalocean.Client(token=self.config['digitalocean']['api_token'])

            while True:
                # Get droplet metrics for the last hour
                now = datetime.utcnow()
                start_time = now - datetime.timedelta(hours=1)

                # Get all droplets
                droplets = client.droplets.list()

                for droplet in droplets:
                    # Get metrics for each droplet
                    metrics = client.droplet_metrics.get(
                        droplet_id=droplet.id,
                        start=start_time.isoformat(),
                        end=now.isoformat()
                    )

                    # Process CPU, memory and disk metrics
                    for metric in metrics:
                        yield {
                            'timestamp': datetime.utcnow(),
                            'source': 'digitalocean',
                            'event_type': 'metric',
                            'message': f"Droplet {droplet.name} - {metric.type}: {metric.value}",
                            'raw_data': {
                                'droplet_id': droplet.id,
                                'droplet_name': droplet.name,
                                'metric_type': metric.type,
                                'metric_value': metric.value,
                                'timestamp': metric.timestamp
                            }
                        }

                await asyncio.sleep(300)  # 5 minutes

        except Exception as e:
            logging.error(f"Error collecting DigitalOcean logs: {e}")
        pass