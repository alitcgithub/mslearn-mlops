from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.core.exceptions import ResourceNotFoundError

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)[cite: 2]
    parser.add_argument("--resource-group", dest="resource_group", required=True)[cite: 2]
    parser.add_argument("--workspace", dest="workspace", required=True)[cite: 2]
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint")[cite: 2]
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")[cite: 2]

    return parser.parse_args()[cite: 2]


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()[cite: 2]
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )[cite: 2]


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        return ml_client.online_endpoints.get(name=endpoint_name)[cite: 2]
    except ResourceNotFoundError:[cite: 2]
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for MLflow diabetes model",
            auth_mode="key",
        )[cite: 2]
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()[cite: 2]


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
) -> ManagedOnlineDeployment:
    model_name = "diabetes-model"
    
    print(f"Retrieving latest version of registered model '{model_name}' from workspace registry...")
    try:
        registered_model = ml_client.models.get(name=model_name, label="latest")
    except Exception:
        registered_model = ml_client.models.get(name=model_name, version="1")
        
    print(f"Using model: {registered_model.name} (Version: {registered_model.version})")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )[cite: 2]

    return ml_client.online_deployments.begin_create_or_update(deployment).result()[cite: 2]


def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)[cite: 2]
    endpoint.traffic = {deployment_name: 100}[cite: 2]
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()[cite: 2]


def main() -> None:
    args = parse_args()[cite: 2]

    print("Connecting to Azure Machine Learning workspace...")[cite: 2]
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,[cite: 2]
        resource_group=args.resource_group,[cite: 2]
        workspace=args.workspace,[cite: 2]
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")[cite: 2]
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)[cite: 2]
    print(f"Using endpoint: {endpoint.name}")[cite: 2]

    print(f"Creating or updating deployment '{args.deployment_name}'...")[cite: 2]
    deployment = create_or_update_deployment(
        ml_client=ml_client,[cite: 2]
        endpoint_name=endpoint.name,[cite: 2]
        deployment_name=args.deployment_name,[cite: 2]
    )
    print(f"Deployment state: {deployment.provisioning_state}")[cite: 2]

    print("Directing 100% of traffic to the deployment...")[cite: 2]
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)[cite: 2]

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)[cite: 2]
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")[cite: 2]


if __name__ == "__main__":
    main()[cite: 2]
