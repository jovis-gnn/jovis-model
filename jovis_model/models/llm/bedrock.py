import json
import boto3


class Bedrock:
    def __init__(self, config):
        self.config = config
        sts_client = boto3.client("sts")
        response = sts_client.assume_role(
            RoleArn=self.config.params.role_arn,
            RoleSessionName=self.config.params.region_name,
        )
        credentials = response["Credentials"]

        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.config.params.region_name,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        self.bedrock_agent_runtime = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=self.config.params.region_name,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    def inference(self, body):
        response = self.bedrock_runtime.invoke_model(
            body=body, modelId=self.config.params.m_type_id
        )
        response_body = json.loads(response.get("body").read())
        response = response_body["content"][0]["text"]

        return response
