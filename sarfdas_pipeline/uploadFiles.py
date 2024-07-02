import os
from google.cloud import storage


def main():
    # Set your Google Cloud project and bucket name
    project_id = 'servir-ee'
    bucket_name = 'bucketsvmk'
    cloud_folder_name = 'khForest'  # Folder name in the bucket

    # Create a client to interact with the storage service
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)

    # Set the local folder path containing the TIF files
    folder_path = '/home/ubuntu/forestAlert/data/'

    # Iterate over the folders
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_dir):
            tif_file_path = os.path.join(folder_dir, f"{folder_name}.tif")

            # Check if the TIF file exists locally
            if os.path.exists(tif_file_path):
                # Check if the TIF file exists in the bucket
                # Set the desired blob name in the bucket
                blob_name = os.path.join(
                    cloud_folder_name, f"{folder_name}.tif")
                blob = bucket.blob(blob_name)

                if not blob.exists():
                    # Upload the TIF file to the bucket
                    blob.upload_from_filename(tif_file_path)
                    print(f"Uploaded {tif_file_path} to {blob_name}")
                else:
                    print(
                        f"{tif_file_path} already exists in the {blob_name} bucket")
            else:
                print(f"{tif_file_path} does not exist locally")
        print("Running uploadFiles")


if __name__ == "__main__":
    main()
