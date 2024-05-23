import json
import boto3


class Bedrock:
    def __init__(self, config):
        self.config = config
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.config.params.region_name,
        )
        self.bedrock_agent_runtime = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=self.config.params.region_name,
        )

    def inference(self, body):
        response = self.bedrock_runtime.invoke_model(
            body=body, modelId=self.config.params.m_type_id
        )
        response_body = json.loads(response.get("body").read())
        response = response_body["content"][0]["text"]

        return response
