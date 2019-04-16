import boto3

lambda_client = boto3.client('lambda')

fn_names = ["alex0", "alex1", "alex2"]
fn_role = "arn:aws:iam::910463673602:role/service-role/serferDemoOne-role-9ecz4s2a"
pytorch_layer = "arn:aws:lambda:us-east-2:910463673602:layer:pytorchv5-p36:1"
redis_layer = "arn:aws:lambda:us-east-2:910463673602:layer:my-python36-redis:1"

for fn_name in fn_names:
    print("Creating " + fn_name)
    lambda_client.create_function(
        FunctionName = fn_name,
        Runtime='python3.6',
        Role=fn_role,
        Handler="{0}.lambda_handler".format(fn_name),
        Code={'ZipFile': open("{0}.zip".format(fn_name), 'rb').read(), },
        VpcConfig={
            'SubnetIds': [
                'subnet-c94430b3',
            ],
            'SecurityGroupIds': [
                'sg-08263df78fdfff7d3',   
            ]
        },
        Layers=[pytorch_layer, redis_layer]
    )
    print("Created " + fn_name)
#response = lambda_client.add_permission(
#    FunctionName=fn_name, 
#    StatementId='2', 
#    Action='lambda:InvokeFunction', 
#    Principal='s3.amazonaws.com', 
#    SourceArn='arn:aws:s3:::serferdemoone', 
#    SourceAccount='910463673602'
#)
