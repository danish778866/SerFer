import boto3
payload=b"""{
"key": "imagelayer0.some"
}"""

client = boto3.client('lambda')
print("Invoking the function")
response = client.invoke(
            FunctionName="alex0",
            InvocationType="Event",
            Payload=payload
        )
print(response)
print("Done")
