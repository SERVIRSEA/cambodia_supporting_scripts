import os
import subprocess
import ee
from datetime import datetime


def main():
    # Authenticate to the Earth Engine servers
    ee.Initialize()

    # Set your Google Cloud Storage bucket name and folder path
    gcs_folder_path = 'gs://bucketsvmk/khForest/'
    # gcs_folder_path = 'gs://bucketsvmk/khAlert/S1A_IW_GRDH_1SDV_20231017T230144_20231017T230209_050811_061FC1_3850.tif'

    # Set the Earth Engine asset id for the image collection
    # ee_asset_id = 'projects/cemis-camp/assets/GPLforestAlerts'
    ee_asset_id = 'projects/cemis-camp/assets/khAlertsv3'

    # Get the list of images currently in the Earth Engine collection
    ee_collection = ee.ImageCollection(ee_asset_id)
    ee_images = [os.path.basename(item['id'])
                 for item in ee_collection.getInfo()['features']]

    # Get the list of images in the Google Cloud Storage bucket
    gcs_ls_command = f"gsutil ls {gcs_folder_path}"
    gcs_images_output = subprocess.check_output(
        gcs_ls_command, shell=True).decode('utf-8')
    # The output ends with a newline, so the last element is an empty string
    gcs_images = gcs_images_output.split('\n')[:-1]

    print(gcs_images)

    # Set the desired band names
    band_names = 'time0,time1,time2'
    # band_names = 'landclass'

    # Iterate over the Google Cloud Storage images
    for gcs_image in gcs_images:
        # Extract the image name from the full Google Cloud Storage path
        image_name = os.path.basename(gcs_image)

        # Ensure the file is a GeoTIFF file
        if image_name.endswith('.tif'):
            # Remove the .tif extension
            image_name_no_ext = os.path.splitext(image_name)[0]

            # Extract the timestamp from the image name and convert it to a UNIX timestamp
            timestamp_str = image_name.split('_')[4]
            date_str = timestamp_str[:8]
            time_str = timestamp_str[9:15]
            date_time_obj = datetime.strptime(
                date_str + time_str, '%Y%m%d%H%M%S')
            unix_timestamp = int(date_time_obj.timestamp() * 1000)

            # If the image is not already in the Earth Engine collection
            if image_name_no_ext not in ee_images:
                # Upload the image to the Earth Engine collection
                print(f"Uploading {image_name}")
                # ee_upload_command = f"earthengine upload image --asset_id={ee_asset_id}/{image_name_no_ext} --bands={band_names} {gcs_image} --nodata_value=-9999 --time_start={unix_timestamp}"
                ee_upload_command = f"earthengine upload image --asset_id={ee_asset_id}/{image_name_no_ext} --bands={band_names} {gcs_image} --nodata_value=-9999 --time_start={unix_timestamp}"
                print(ee_upload_command)
                subprocess.call(ee_upload_command, shell=True)
            else:
                print(f"{image_name} already exists in the Earth Engine collection")
        else:
            print(f"Skipping {image_name} as it is not a .tif file")
        print("Running uploadGEE")
