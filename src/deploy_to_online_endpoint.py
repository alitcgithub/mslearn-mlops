import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.identity import DefaultAzureCredential

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, default="blue")
    args = parser.parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Online endpoint for diabetes model deployment",
        auth_mode="key",
    )
    
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Using endpoint: {args.endpoint_name}")
    except Exception as e:
        print(f"Error creating/updating endpoint: {e}")
        raise e

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client,
        args.endpoint_name,
        args.deployment_name,
    )
    print("Deployment complete.")

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
        instance_type="Standard_F2s_v2",
        instance_count=1,
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()

if __name__ == "__main__":
    main()
