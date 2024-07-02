import ee
from datetime import date, timedelta
import sys


def main(year, start_week):
    print(
        f"Running sarAlertBiWeekly for year {year}, start week {start_week}")
    # Authenticate to the Earth Engine servers
    ee.Initialize()

    def get_biweekly_dates(year, latest_week):
        # Calculate the start week
        start_week = latest_week

        # Get the start date of the start week
        start_date = date(year, 1, 1)
        if start_date.weekday() > 0:  # If the year doesn't start on Monday
            start_date += timedelta(days=7 - start_date.weekday())
        start_date += timedelta(weeks=start_week - 1)

        # Ensure the start date is within the given year
        if start_date.year < year:
            start_date = date(year, 1, 1)

        # Calculate the end date (2 weeks from start date minus one day)
        end_date = start_date + timedelta(weeks=2) - timedelta(days=1)

        # Convert to ee.Date format
        start_date_ee = ee.Date.fromYMD(
            start_date.year, start_date.month, start_date.day)
        end_date_ee = ee.Date.fromYMD(
            end_date.year, end_date.month, end_date.day)

        return start_date_ee, end_date_ee

    def mask_with_forest(input_image, forest_layer):
        """
        Masks the input image with the forest layer.

        Parameters:
        input_image (ee.Image): The image to be masked.
        forest_layer (ee.Image): The forest layer to use as a mask.

        Returns:
        ee.Image: The masked image.
        """
        # The forest layer should be a binary image where forested areas are 1 (or true) and non-forested areas are 0 (or false)
        # Update the mask of the input image with the forest layer
        masked_image = input_image.updateMask(forest_layer)

        return masked_image

    def transform_image(image, year, week_number):

        # Define Sentinel-1 collection and filter by date and area
        def get_sentinel1_footprint(image):
            # Image has a spatial reference to filter Sentinel-1 data
            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
            footprint = sentinel1.filterDate(image.date().advance(-1, 'day'), image.date(
            ).advance(1, 'day')).filterBounds(image.geometry()).first().geometry()
            return footprint

        # Construct the image paths using the year variable.
        imageTCC = ee.Image(f"projects/servir-mekong/UMD/TCC_C02/{year - 1}")
        imageTCH = ee.Image(f"projects/servir-mekong/UMD/TCH_C02/{year - 1}")

        # Create forest cover map
        forestLayer = imageTCC.select('b1').gt(
            10).And(imageTCH.select('b1').gt(5))

        image = ee.Image(image)

        date_alert = ee.Date(image.get('system:time_start'))
        start_date, end_date = get_biweekly_dates(year, week_number)

        day_of_year = date_alert.getRelative('day', 'year')

        # Define the alert only where probability > 50 in 3 time data.
        condition = image.select('time0').gt(50) \
            .And(image.select('time1').gt(50)) \
            .And(image.select('time2').gt(50))

        # Create an alert image with day_of_year values where the condition is met
        # Create the alert band name based on the year
        alert_band_name = f'alertdate{year % 100}'
        alert_image = ee.Image.constant(day_of_year).updateMask(
            condition).rename(alert_band_name).int16()

        sar_forest_filtered = mask_with_forest(
            alert_image, forestLayer).int16()

        return sar_forest_filtered.set({
            'system:time_start': start_date,
            'system:time_end': end_date
        })

    def process_biweekly_period(year, week_number, sarCollection, aoiKhADM0):
        start_date, end_date = get_biweekly_dates(year, week_number)

        print(start_date.millis(), end_date.millis())

        filtered_collection = sarCollection.filterDate(start_date, end_date)

        sarBiweekly = filtered_collection.map(
            lambda image: transform_image(image, year, week_number))

        # Combine the images in the collection using mosaic
        mosaic_image = sarBiweekly.mosaic().set({
            'system:time_start': start_date.millis(),
            'system:time_end': end_date.millis()
        })

        # Create the name for the image
        imageName = "khm_sar_alert_img_{}_{}".format(
            year, f"{week_number:02d}")

        # Define the export task
        export_task = ee.batch.Export.image.toAsset(
            image=mosaic_image,
            description='merged_khAlertsbiWeekly',
            assetId=output_asset + imageName,
            scale=10,
            maxPixels=1e13,
            region=aoiKhADM0.geometry()
        )

        # Start the export task
        export_task.start()

        # Print a confirmation message
        print("Export started for week number:", week_number)

    output_asset = "projects/cemis-camp/assets/khForestAlert/sarBiWeeklyAlert/"

    aoiKhADM0 = ee.FeatureCollection("projects/servir-mekong/admin/KHM_adm0")

    original_collection = ee.ImageCollection(
        "projects/cemis-camp/assets/khAlertsv3")

    process_biweekly_period(year, start_week, original_collection, aoiKhADM0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sarAlertBiWeekly.py <year> <start_week>")
        sys.exit(1)

    year = int(sys.argv[1])
    start_week = int(sys.argv[2])
    main(year, start_week)
