import os
import io
import requests
import numpy as np
from osgeo import gdal
import google.auth
import ee
from retrying import retry
import model
from typing import Tuple
import sys
import subprocess
#import geemap
from google.api_core import exceptions
import concurrent.futures
from numpy.lib.recfunctions import structured_to_unstructured
from tensorflow.keras import backend as K

import glob
import osr
import rasterio
from rasterio.merge import merge
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import argparse
import tensorflow as tf
from tqdm import tqdm
import argparse

#import matplotlib.pyplot as plt
ee.Initialize()
max_workers = 2
nDays = 12
# Initialize list to store output file names
output_files = []

# Set environment variable to disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class EarthEngine:
    def __init__(self):
        # Define the scale to convert meters to pixels
        self.SCALE = 10

    # Define a custom error handler function that does nothing
    def handleError(err_class, err_num, err_msg):
        pass

    # Push the custom error handler onto the GDAL error handler stack
    gdal.PushErrorHandler(handleError)

    def init(self) -> None:
        """Authenticate and initialize Earth Engine with the default credentials."""
        # Use the Earth Engine High Volume endpoint.
        # https://developers.google.com/earth-engine/cloud/highvolume
        credentials, project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/earthengine",
            ]
        )
        ee.Initialize(
            credentials.with_quota_project(None),
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_delay=600000)  # milliseconds
    def get_patch(self, image: ee.Image, lonlat: Tuple[float, float], patch_size: int, scale: int) -> np.ndarray:
        """Fetches a patch of pixels from Earth Engine.
        It retries if we get error "429: Too Many Requests".
        Args:
            image: Image to get the patch from.
            lonlat: A (longitude, latitude) pair for the point of interest.
            patch_size: Size in pixels of the surrounding square patch.
            scale: Number of meters per pixel.
        Raises:
            requests.exceptions.RequestException
        Returns: The requested patch of pixels as a NumPy array with shape (width, height, bands).
        """
        point = ee.Geometry.Point(lonlat)

        # Create the download URL for the patch
        url = image.getDownloadURL(
            {
                "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
                "dimensions": [patch_size, patch_size],
                "format": "NPY",
            }
        )

        # If we get "429: Too Many Requests" errors, it's safe to retry the request.
        # The Retry library only works with `google.api_core` exceptions.
        response = requests.get(url)
        if response.status_code == 429:
            raise exceptions.TooManyRequests(response.text)

        # Still raise any other exceptions to make sure we got valid data.
        response.raise_for_status()

        # Parse the response content as a NumPy array
        return np.load(io.BytesIO(response.content), allow_pickle=True)

class ImageProcessor:
    def __init__(self, image_id):
        self.image_id = image_id

    def getImage(self):
        """
        Retrieve a Sentinel-1 image and pre-process it for further analysis.

        Args:
            item (str): The item index of the image to retrieve.

        Returns:
            Tuple: A tuple containing the processed image and its geometry.
        """
        # Set mode and bands for the image collection
        MODE = 'DESCENDING'
        bands = ['VH_after0', 'VH_before0', 'VH_before1', 'VH_before2', 'VH_before3', 'VH_mean', 'VH_std', 'VV_after0', 'VV_before0', 'VV_before1', 'VV_before2','VV_before3','VV_mean', 'VV_std']

        # Import Sentinel-1 Collection
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.eq('orbitProperties_pass', MODE))\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\

        # Retrieve the specified image
        img = ee.Image(s1.filter(ee.Filter.eq("system:index",self.image_id)).first())

        # Extract the image geometry and properties
        geom = img.geometry()
        prop = img.toDictionary()
        name = img.get("system:index").getInfo()
        timeStamp = img.get("system:time_start")
        date = ee.Date(timeStamp)

        # Import median and standard deviation images for the Mekong region
        med = ee.Image("projects/servir-mekong/sentinel1/medMekong").select(["VV","VH"],["VV_mean","VH_mean"]) #.divide(10000)
        std = ee.Image("projects/servir-mekong/sentinel1/stdMekongInt").select(["VV_stdDev","VH_stdDev"],["VV_std","VH_std"]) #.divide(10000)

        # Create a filtered image collection for the specified image and its temporal series
        s1Item = s1.filterBounds(geom)
        beforeSeries = self.create_series_before(s1Item,date.advance(-nDays,"days"))

        # Select the after0 band and add it to the temporal series with the median and standard deviation images
        after0 = img.select(["VV","VH"],["VV_after0","VH_after0"]).toFloat()
        image = beforeSeries.addBands(after0).addBands(med).addBands(std)

        # Select and return the desired bands, set data type and fill gaps
        image = image.select(bands).toFloat().unmask(0,False)
        return image, geom

    def add_ratio(self, img):
        """
        Add a ratio band to an input image.

        Args:
            img (ee.Image): The input image to add the ratio band to.

        Returns:
            ee.Image: The input image with a new ratio band.
        """
        geom = img.geometry()
        vv = self.to_natural(img.select(['VV'])).rename(['VV'])
        vh = self.to_natural(img.select(['VH'])).rename(['VH'])
        ratio = vh.divide(vv).rename(['ratio'])
        return ee.Image(ee.Image.cat(vv, vh, ratio).copyProperties(img, ['system:time_start'])).clip(geom).copyProperties(img)

    def erode_geometry(self, image):
        """
        Erode the geometry of an input image.

        Args:
            image (ee.Image): The input image to erode.

        Returns:
            ee.Image: The input image with eroded geometry.
        """
        return image.clip(image.geometry().buffer(-1000))

    def to_natural(self, img):
        """
        Convert an input image from dB to natural.

        Args:
            img (ee.Image): The input image to convert.

        Returns:
            ee.Image: The input image in natural scale.
        """
        return ee.Image(10.0).pow(img.select(0).divide(10.0))

    def to_db(self, img):
        """
        Convert an input image from natural to dB.

        Args:
            img (ee.Image): The input image to convert.

        Returns:
            ee.Image: The input image in dB scale.
        """
        return ee.Image(img).log10().multiply(10.0)

    def create_series_before(self, collection, date, iters=4, nday=12):
        """
        Creates a time series of images with the mean of the VH and VV polarization bands for a period of time
        before the given date.

        Args:
            collection (ee.ImageCollection): The input image collection to create the time series from.
            date (ee.Date): The date to create the time series around.
            iters (int): The number of iterations to make. Default is 4.
            nday (int): The number of days in each iteration. Default is 12.

        Returns:
            ee.Image: An image with bands representing the mean of the VH and VV polarization bands for each iteration
            before the given date.
        """
        iterations = list(range(1, iters * nday, nday))
        names = list(["_before{:01d}".format(x) for x in range(0, iters, 1)])

        def return_collection(day, name):
            start = ee.Date(date).advance(-day, "days").advance(-nday, "days")
            end = ee.Date(date).advance(-day, "days")
            band_names = ["VV" + name, "VH" + name]
            return ee.Image(collection.filterDate(start, end).mean()) \
                .select(["VV", "VH"], band_names) \
                .set("system:time_start", start)

        return self.toBands(ee.ImageCollection.fromImages(list(map(return_collection, iterations, names))))

    def createSeriesAfter(self, collection, date, iters=2, nday=12):
        """Create a time series of images from a collection for a given date and time range after the date.

        Args:
            collection (ee.ImageCollection): The image collection to create the time series from.
            date (ee.Date): The date to create the time series for.
            iters (int, optional): The number of iterations to create. Defaults to 2.
            nday (int, optional): The number of days in each iteration. Defaults to 12.

        Returns:
            ee.Image: The stacked image containing the time series.
        """

        # Create iterations and names for the bands
        iterations = list(range(1, iters * nday, nday))
        names = list(["_after{:01d}".format(x) for x in range(0, iters, 1)])

        def returnCollection(day, name):
            # Define start and end dates
            start = ee.Date(date).advance(day, "days")
            end = ee.Date(date).advance(day, "days").advance(nday, "days")

            # Define band names
            bandNames = ["VV" + name, "VH" + name]

            # Get the mean image for the time range and select bands
            return ee.Image(collection.filterDate(start, end).mean()) \
                .select(["VV", "VH"], bandNames) \
                .set("system:time_start", start)

        # Create the time series by mapping the function over the iterations and names
        return self.toBands(ee.ImageCollection.fromImages(list(map(returnCollection, iterations, names))))


    def toBands(self, collection):
        """Stack an image collection into a single image with bands corresponding to the images in the collection.

        Args:
            collection (ee.ImageCollection): The image collection to stack.

        Returns:
            ee.Image: The stacked image.
        """

        def createStack(img, prev):
            # Add the current image as a new band to the previous image
            return ee.Image(prev).addBands(img)

        # Stack the images in the collection using the createStack function and the iterate method
        stack = ee.Image(collection.iterate(createStack, ee.Image(1)))

        # Select all the bands except the first one, which is just an empty band from the initialization
        stack = stack.select(ee.List.sequence(1, stack.bandNames().size().subtract(1)))

        return stack



class TileProcessor:
    def __init__(self,earth_engine, image, scale, patch_size, overlap_size, myModel):
        self.image = image
        self.scale = scale
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.myModel = myModel
        self.outdir = outdir

    def rescale(self, arr):
        rescaled_arr = np.zeros_like(arr)
        num_channels = arr.shape[-1]
        
        for i in range(num_channels):
            min_val = np.min(arr[..., i])
            max_val = np.max(arr[..., i])
            rescaled_arr[..., i] = (arr[..., i] - min_val) / (max_val - min_val)
            
        return rescaled_arr

    def load_data(self, patch):
        # Rescale the patch
        patch = self.rescale(patch)

        return patch

    def slice_into_patches(self, image):
        # Add batch and channel dimensions to the image
        image = tf.expand_dims(image, 0)
        image = tf.expand_dims(image, -1)

        # Define the sizes and strides for the patches
        sizes = [1, self.patch_size, self.patch_size, 1]
        strides = [1, self.stride, self.stride, 1] 
        rates = [1, 1, 1, 1]

        # Use tf.image.extract_patches to extract the patches
        patches = tf.image.extract_patches(image, sizes, strides, rates, 'VALID')

        # Reshape the patches to have shape (num_patches, patch_size, patch_size, 1)
        _, height, width, _ = patches.shape
        num_patches = height * width
        patches = tf.reshape(patches, (num_patches, self.patch_size, self.patch_size, 1))

        return patches

    def create_tf_dataset(self, patches):
        # Initialize the dataset with your data
        dataset = tf.data.Dataset.from_tensor_slices(patches)

        # Apply the 'load_data' function to every element in the dataset
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def process_patch(self, patch):
        # ... pre-process the patch ...

        # Make a prediction
        probabilities = self.myModel.predict(patch,verbose=0)
        K.clear_session()
        return probabilities[0]

    def process_in_patches(self, input_data, patch_size, stride):
        _, height, width, num_channels = input_data.shape
        input_data = np.squeeze(input_data, axis=0)

        input_tensor = tf.constant(input_data)

        patches = tf.image.extract_patches(
            images=tf.expand_dims(input_tensor, 0),
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, (-1, patch_size, patch_size, num_channels))
        patches = tf.map_fn(self.rescale, patches)

        probabilities = self.myModel(patches) #,verbose=0)

        patch_rows = (height - patch_size) // stride + 1
        patch_cols = (width - patch_size) // stride + 1

        results = np.zeros((height, width, 2))
        counts = np.zeros((height, width, 2))

        for i in range(patch_rows):
            for j in range(patch_cols):
                patch_prob = probabilities[i * patch_cols + j]
                results[i * stride:i * stride + patch_size, j * stride:j * stride + patch_size, :] += patch_prob
                counts[i * stride:i * stride + patch_size, j * stride:j * stride + patch_size, :] += 1

        counts = np.where(counts == 0, 1e-10, counts)
        # Normalize by counts to get average probabilities when there's overlap
        results = results / counts
        return results

    def getTiles(self, counter, x, y, xl, yl):
        #counter, x, y, xl, yl = inputs
        overlap = int(self.overlap_size)
        pixel_size = self.scale / 111320

        # Calculate the patch boundaries
        xmin, ymin, xmax, ymax = x, y, x + (overlap * pixel_size), y + (overlap * pixel_size)
        lon, lat = (xmin + xmax) / 2, (ymin + ymax) / 2
        lonlat = (lon, lat)

        point = ee.Geometry.Point(lonlat)
        geom = point.buffer(self.scale * self.patch_size / 2, 1).bounds(1)
        coords = np.array(geom.getInfo().get("coordinates"))[0]

        xmin, ymin, xmax, ymax = coords[0][0] + (overlap * pixel_size), coords[0][1] + (overlap * pixel_size), coords[1][0] - (overlap * pixel_size), coords[2][1] - (overlap * pixel_size)

        coords = [xmin, ymin, xmax, ymax]

        imVH = image.select(["VH_after0","VH_before0","VH_before1","VH_before2","VH_before3","VH_mean","VH_std"])
        patchVH = earth_engine.get_patch(imVH, lonlat, self.patch_size, self.scale)
        inputsVH = structured_to_unstructured(patchVH)
        input_dataVH = np.stack([inputsVH])

        imVV = image.select(["VV_after0","VV_before0","VV_before1","VV_before2","VV_before3","VV_mean","VV_std"])
        patchVV = earth_engine.get_patch(imVV, lonlat, self.patch_size, self.scale)
        inputsVV = structured_to_unstructured(patchVV)
        input_dataVV = np.stack([inputsVV])


        VH_after0 =np.power(10.0, np.divide(input_dataVH[:,:,:,0], 10.0))
        VH_before0 =np.power(10.0, np.divide(input_dataVH[:,:,:,1], 10.0))
        VH_before1=np.power(10.0, np.divide(input_dataVH[:,:,:,2], 10.0))
        VH_before2=np.power(10.0, np.divide(input_dataVH[:,:,:,3], 10.0))
        VH_before3=np.power(10.0, np.divide(input_dataVH[:,:,:,4], 10.0))
        VH_mean=input_dataVH[:,:,:,5] / 10000
        VH_std=input_dataVH[:,:,:,6] / 10000
   
        
        VV_after0 =np.power(10.0, np.divide(input_dataVV[:,:,:,0], 10.0))
        VV_before0 =np.power(10.0, np.divide(input_dataVV[:,:,:,1], 10.0))
        VV_before1=np.power(10.0, np.divide(input_dataVV[:,:,:,2], 10.0))
        VV_before2=np.power(10.0, np.divide(input_dataVV[:,:,:,3], 10.0))
        VV_before3=np.power(10.0, np.divide(input_dataVV[:,:,:,4], 10.0))
        VV_mean=input_dataVV[:,:,:,5] / 10000
        VV_std=input_dataVV[:,:,:,6] / 10000
   
        #VV_after0=np.power(10.0, np.divide(input_data[:,:,:,7], 10.0))
        #VV_before0=np.power(10.0, np.divide(input_data[:,:,:,8], 10.0))
        #VV_before1=np.power(10.0, np.divide(input_data[:,:,:,9], 10.0))
        #VV_before2=np.power(10.0, np.divide(input_data[:,:,:,10], 10.0))
        #VV_before3=np.power(10.0, np.divide(input_data[:,:,:,11], 10.0))
        #VV_mean= input_data[:,:,:,12] / 10000
        #VV_std= input_data[:,:,:,13] / 10000

        time0 = np.stack((VH_after0, VH_before0, VH_before1, VH_mean, VH_std, VV_after0, VV_before0,VV_before1, VV_mean, VV_std), axis=-1)
        time1 = np.stack((VH_after0, VH_before1, VH_before2, VH_mean, VH_std, VV_after0, VV_before1,VV_before2, VV_mean, VV_std), axis=-1)
        time2 = np.stack((VH_after0, VH_before2, VH_before3, VH_mean, VH_std, VV_after0, VV_before2,VV_before3, VV_mean, VV_std), axis=-1)

        #probabilities0 = myModel.predict(time0)[0]
        #probabilities1 = myModel.predict(time1)[0]
        #probabilities2 = myModel.predict(time2)[0]
        probabilities0 = self.process_in_patches(time0,128,96)
        probabilities1 = self.process_in_patches(time1,128,96)
        probabilities2 = self.process_in_patches(time2,128,96)



        # Get the predictions from the probability distributions.
        predictions0 =   np.squeeze(probabilities0)
        predictions1 =   np.squeeze(probabilities1)
        predictions2 =   np.squeeze(probabilities2)

        # Create a numpy array of data
        predictions0 = predictions0 * 100
        predictions1 = predictions1 * 100
        predictions2 = predictions2 * 100

        alert0 = predictions0[overlap:-overlap, overlap:-overlap][:,:,0]
        alert1 = predictions1[overlap:-overlap, overlap:-overlap][:,:,0]
        alert2 = predictions2[overlap:-overlap, overlap:-overlap][:,:,0]

        alert = np.mean([alert0, alert1, alert2], axis=0)
        return alert0, alert1, alert2, coords


class GridMaker:
    def __init__(self, geom, scale, patch_size, overlap_size):
        self.geom = geom
        self.scale = scale
        self.patch_size = patch_size
        self.overlap_size = overlap_size

    def make_grid(self):
        """
        Create a grid of points within a given geometry.

        Returns:
            ee.FeatureCollection: A FeatureCollection containing the grid points.
        """

        # Get the coordinates of the geometry bounds
        coords = np.array(self.geom.bounds().getInfo().get("coordinates"))[0]

        xmin = coords[0][0]
        xmax = coords[1][0]
        ymin = coords[0][1]
        ymax = coords[2][1]

        # Convert scale to degrees
        pixel_size = self.scale / 111320

        # Calculate the number of patches needed
        n_patches_x = int((xmax - xmin) / ((self.patch_size - (self.overlap_size * 2)) * pixel_size))
        n_patches_y = int((ymax - ymin) / ((self.patch_size - (self.overlap_size * 2)) * pixel_size))

        num_patches = (n_patches_x + 1) * (n_patches_y + 1)
        print("Number of patches needed: {}".format(num_patches))

        # Create the grid ranges
        y_range = np.arange(ymin, ymax, (self.patch_size - (self.overlap_size * 2)) * pixel_size)
        x_range = np.arange(xmin, xmax, (self.patch_size - (self.overlap_size * 2)) * pixel_size)

        # Create a list of features representing the grid points
        fc = []
        for yl, y in enumerate(y_range):
            for xl, x in enumerate(x_range):
                # Calculate the patch boundaries
                xmin = x
                xmax = x + (self.overlap_size * pixel_size)
                ymin = y
                ymax = y + (self.overlap_size * pixel_size)

                lon = (xmin + xmax) / 2
                lat = (ymin + ymax) / 2
                lonlat = (lon, lat)
                geo = ee.Geometry.Point(lonlat)
                geo = geo.buffer(self.scale * self.patch_size / 2, 1).bounds(1)
                point = ee.Feature(None).setGeometry(geo).set("yl", yl).set("xl", xl)
                fc.append(point)

        # Convert the list of features into a FeatureCollection and filter by the input geometry
        fc = ee.FeatureCollection(fc).filterBounds(self.geom)

        # Define a function to get the coordinates of each point
        def get_coords(feature):
            # Get the geometry of the feature
            geo = feature.geometry().centroid(1)
            xl = feature.get("xl")
            yl = feature.get("yl")
            return ee.Feature(None).setGeometry(geo).set("yl", yl).set("xl", xl)

        # Map the function over the feature collection
        fc_with_coords = fc.map(get_coords)
        return fc_with_coords.getInfo().get("features")



class RasterWriter:
    def __init__(self, patch_size, overlap_size):
        self.patch_size = patch_size
        self.overlap_size = overlap_size

    def write_output(self, counter, raster0, raster1, raster2, out_file, coords):
        """
        Create a new GeoTIFF file with the given filename and writes the given data into it.

        Args:
            counter (int): A counter to create a unique filename for each output file.
            raster0 (ndarray): The data to be written to the first band of the output file.
            raster1 (ndarray): The data to be written to the second band of the output file.
            raster2 (ndarray): The data to be written to the third band of the output file.
            outdir (str): The output directory where the file will be created.
            coords (list): A list of coordinates specifying the extent of the output file.

        Returns:
            None
        """
        xmin = coords[0]
        ymin = coords[1]
        xmax = coords[2]
        ymax = coords[3]

        # Create the output filename
        #out_file = outdir + "/" + str(counter).zfill(4) + ".tif"

        # Create a new GDAL driver for GeoTIFF
        driver = gdal.GetDriverByName("GTiff")

        # Create the output raster file
        out_raster = driver.Create(out_file, self.patch_size - self.overlap_size - self.overlap_size,
                                    self.patch_size - self.overlap_size - self.overlap_size, 3,gdal.GDT_Int16)

        # Set the spatial reference system (SRS)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        out_raster.SetProjection(srs.ExportToWkt())

        # Set the extent of the file
        out_raster.SetGeoTransform((xmin, (xmax - xmin) / (self.patch_size - self.overlap_size - self.overlap_size),
                                     0, ymax, 0, -(ymax - ymin) / (self.patch_size - self.overlap_size - self.overlap_size)))

        # Write the data to the file
        out_raster.GetRasterBand(1).WriteArray(raster0)
        out_raster.GetRasterBand(2).WriteArray(raster1)
        out_raster.GetRasterBand(3).WriteArray(raster2)

        # Close the file
        out_raster = None


class TaskRunner:
    def __init__(self, earth_engine, image, scale, patch_size, overlap_size, myModel, outdir):
        self.earth_engine = earth_engine
        self.image = image
        self.scale = scale
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.myModel = myModel
        self.outdir = outdir
        self.tile_processor = TileProcessor(earth_engine, image, scale, patch_size, overlap_size, myModel)
        self.writer = RasterWriter(patch_size,overlap_size)


        # Create the output directory if it doesn't existTaskRunner
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def run_task(self, counter, x, y, xl, yl):
        out_file = f"{self.outdir}/{str(counter).zfill(4)}.tif"

        #ile_processor = TileProcessor(self.earth_engine, self.image, self.scale, self.patch_size, self.overlap_size, self.myModel)
        raster0, raster1, raster2, coords = self.tile_processor.getTiles(counter, x, y, xl, yl)

        #writer = RasterWriter(self.patch_size,self.overlap_size)
        self.writer.write_output(counter, raster0, raster1, raster2, out_file, coords)

        # Check if the file was created successfully
        if not os.path.exists(out_file):
            raise Exception(f"Error: output file {out_file} was not created successfully. Restarting task.")

        #print(f"Task {counter} completed successfully.")


class GeoTiffMerger:
    def __init__(self):
        pass

    def merge_geotiffs(self, folder_path, image_id):
        print("Starting to merge GeoTIFFs...")
        geotiffs = glob.glob(os.path.join(folder_path, "*.tif"))

        # Read the GeoTIFFs and merge them
        src_files_to_mosaic = [rasterio.open(tif) for tif in geotiffs]
        try:
            merged_data, merged_transform = merge(src_files_to_mosaic)
            print("GeoTIFFs merged successfully.")

            # Update the metadata of the merged GeoTIFF
            profile = src_files_to_mosaic[0].profile
            profile.update({"driver": "GTiff", "height": merged_data.shape[1], "width": merged_data.shape[2], "transform": merged_transform})

            # Save the merged GeoTIFF to a temporary file
            merged_geotiff_path = os.path.join(folder_path, "merged_geotiff.tif")
            with rasterio.open(merged_geotiff_path, "w", **profile) as dest:
                dest.write(merged_data)
                print("Merged GeoTIFF saved successfully.")
        finally:
            # Always close the open GeoTIFF files
            for src in src_files_to_mosaic:
                src.close()

        return merged_geotiff_path

    def translate_to_cog(self, merged_geotiff_path, folder_path, image_id):
        # Create a Cloud Optimized GeoTIFF from the merged GeoTIFF
        print("Starting to translate to COG...")
        cog_path = os.path.join(folder_path, image_id + ".tif")

        # Add a timeout to the cogeo create function call
        cmd = ["rio", "cogeo", "create", merged_geotiff_path, cog_path]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            print("Translated to COG successfully.")
        except subprocess.TimeoutExpired:
            print("cogeo create function took longer than 10 minutes and was stopped.")
        except subprocess.CalledProcessError as e:
            print(f"cogeo create function returned non-zero exit code {e.returncode} and error message: {e.stderr.decode()}")

        # Remove the temporary merged file
        if os.path.exists(merged_geotiff_path):
            os.remove(merged_geotiff_path)
            print("Temporary merged file removed.")

    def process_geotiffs(self, folder_path, image_id):
        print("Starting to process GeoTIFFs...")
        merged_path = self.merge_geotiffs(folder_path, image_id)
        print("starting converting to cog.")
        self.translate_to_cog(merged_path, folder_path, image_id)
        print("Finished processing GeoTIFFs.")




if __name__ == '__main__':
    """
    This is the main section of the code. It loads the pre-trained model and processes a Sentinel-1 image to
    detect alerts.
    """
    # Define the command-line arguments
    #parser = argparse.ArgumentParser(description='Process Sentinel-1 images.')
    #parser.add_argument('image_id', type=str, help='The ID of the Sentinel-1 image')
    # Parse the arguments
    #args = parser.parse_args()
    # Get the image ID from the command-line arguments
    #image_id = "S1_GRD/S1A_IW_GRDH_1SDV_20230226T225315_20230226T225340_047413_05B117_3768"

    scale = 10
    patch_size = 1152
    overlap_size = 64
    image_id = os.environ.get('IMAGE_ID')

    outdir = r"/root/data/" + image_id
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # Load the pre-trained model weights
    myModel = model.build_efficientNet()
    myModel.load_weights(r"/root/model.h5", by_name=True, skip_mismatch=True)

    # Set the parameters for processing the Sentinel-1 image

    # create image_processor instance
    image_processor = ImageProcessor(image_id)
    # Create a RasterWriter instance
    writer = RasterWriter(patch_size=patch_size, overlap_size=overlap_size)
    # create earthengine instance
    earth_engine = EarthEngine()
    # create merger instance
    geotiff_merger = GeoTiffMerger()



    # Call the getImage() method to get the image and its geometry
    image, geometry = image_processor.getImage()

    print(image.bandNames().getInfo())
    # Create an instance of the GridMaker class

    grid_maker = GridMaker(geometry, scale, patch_size, overlap_size)

    # Call the make_grid method
    grid = grid_maker.make_grid()

    # Create a list to hold the futures returned by the executor
    futures = []

    # Create the task runner
    task_runner = TaskRunner(earth_engine, image, scale, patch_size, overlap_size, myModel, outdir)
    #grid = grid[:10]
    #print(grid)

    # Define the function to be executed in parallel
    def process_patch(counter, patch):
        x = patch.get('geometry').get('coordinates')[0]
        y = patch.get('geometry').get('coordinates')[1]
        xl = patch.get('properties').get('xl')
        yl = patch.get('properties').get('yl')

        out_file = f"{outdir}/{str(counter).zfill(4)}.tif"
        output_files.append(out_file)

        # Check if the output file already exists
        if os.path.exists(out_file):
            print(f"Output file {out_file} already exists. Skipping task {counter}.")
            #return counter
            return "success"

        task_runner.run_task(counter, x, y, xl, yl)

        #return counter, patch
        return "success"

    #for counter, patch in enumerate(grid):
    #    process_patch(counter, patch)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit the tasks to the thread pool
        futures = {executor.submit(process_patch, counter, patch): (counter, patch) for counter, patch in enumerate(grid)}

        completed_tasks = set()
        with tqdm(total=len(grid)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                # Get the result of the future (if any)
                counter, patch = futures[future]
                result = future.result()

                # Check if the result is not "success" (which indicates an error occurred)
                if result != "success":
                    # Resubmit the task if it has not already completed
                    counter, patch = futures[future]
                    if counter not in completed_tasks:
                        future = executor.submit(process_patch, counter, patch)
                        futures[future] = (counter, patch)
                else:
                    # Convert the dictionary to a tuple of (key, value) pairs
                    completed_tasks.add(counter)

                pbar.update(1)  
    
    # Merge the GeoTIFFs after all tasks have completed
    geotiff_merger.process_geotiffs(outdir,image_id)

# Remove the output files
for f in output_files:
    os.remove(f)
