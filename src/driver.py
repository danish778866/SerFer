import boto3

s3_client = boto3.client('s3')

def split_input_img(img_path):
    """
        The user given Query, i.e. Image is stored at img_path
        This image needs to be split (Data Parallelism) for parallel processing
        by Lambda Layers.
    """

def merge_images(merge_order, file_prefix, layer_number):

def upload_file_to_s3(file_path, bucket_name, key_name):
    s3_client.upload_file(file_path,bucket_name, key_name)

def download_file_from_s3(file_path, bucket_name, key_name):
    s3_client.download_file(bucket_name, key_name, file_path)

def poll_for_merge(bucket_name, file_prefix, num_files):
    response = s3_client.list_objects(Bucket=bucket_name, Prefix=file_prefix)
    merge = False
    intermediate_files = []
    if len(response['Contents']) == num_files:
        merge = True
        intermediate_files = response['Contents']
    return merge, intermediate_files

def main():
    """
        This is the main driver function. The driver executes the following steps:
            1. Get the user query, i.e. Image.
            2. Decide the flow of computation.
            3. Split the input image and upload to S3 (This triggers Lambda Layer 1).
            4. Poll S3 bucket to see if all intermediate files are ready for merge.
            5. Merge intermediate files.
            6. Split the merged file for next Lambda layer and upload to S3.
            7. Repeat steps 4 - 7 till the end of the network.
    """

if __name__ == '__main__':
    main()

