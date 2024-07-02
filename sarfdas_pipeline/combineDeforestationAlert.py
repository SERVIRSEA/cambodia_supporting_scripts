import ee
from datetime import date, timedelta
import sys


def main(year, biweekly_period):
    print(
        f"Running combineDeforestationAlert for year {year} and biweekly period {biweekly_period}")

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

    def process_biweekly_period(year, weekNumber, saralert, glad, aoi, collection_id):
        # Get the start and end dates for the biweekly period
        weekNumber = int(weekNumber)
        start_date, end_date = get_biweekly_dates(year, weekNumber)

        # Convert the dates to day of year
        startDoy = start_date.getRelative('day', 'year')
        endDoy = end_date.getRelative('day', 'year')

        print(startDoy, endDoy)

        # Construct the image paths using the year variable.
        imageTCC = ee.Image(f"projects/servir-mekong/UMD/TCC_C02/{year - 1}")
        imageTCH = ee.Image(f"projects/servir-mekong/UMD/TCH_C02/{year - 1}")

        # Create forest cover map
        forestLayer = imageTCC.select('b1').gt(
            10).And(imageTCH.select('b1').gt(5))

        # Convert the processed SAR alerts to an image and rename the band
        processedSarAlerts = saralert.updateMask(saralert.gt(
            startDoy).And(saralert.lt(endDoy))).rename("sar")
        processedSarAlerts_filtered = mask_with_forest(
            processedSarAlerts, forestLayer)

        # Process the GLAD alerts for the given period
        # currentGlad = glad.gt(startDoy).And(glad.lt(endDoy)).unmask(0).rename("glad")
        currentGlad = glad.updateMask(
            glad.gt(startDoy).And(glad.lt(endDoy))).rename("glad")
        currentGlad_filtered = mask_with_forest(currentGlad, forestLayer)

        # Combine the alerts into a single image
        # combinedAlerts = currentGlad_filtered.add(currentTropico_filtered).add(processedSarAlerts_filtered).rename("combinedAlerts")

        # Ensure all images have the same band name before combining
        processedSarAlerts_filtered = processedSarAlerts_filtered.rename(
            "alert")
        currentGlad_filtered = currentGlad_filtered.rename("alert")
        # currentTropico_filtered = currentTropico_filtered.rename("alert")

        # Cast bands to integer type before combining
        processedSarAlerts_int = processedSarAlerts_filtered.toInt()
        currentGlad_int = currentGlad_filtered.toInt()
        # currentTropico_int = currentTropico_filtered.toInt()

        # Combine the alerts into a single image and get the minimum value across the bands
        combinedAlerts_min = ee.ImageCollection(
            [processedSarAlerts_int, currentGlad_int]).min()

        # Create a binary image where pixel values greater than 0 are set to 1
        combinedAlerts = combinedAlerts_min.gt(0).toInt()

        # Add the individual alert bands to the combined alerts image
        alertImageWithBands = combinedAlerts.addBands([
            currentGlad_filtered,
            processedSarAlerts_filtered
        ])

        # Set the properties for the image
        alertImageWithBands = alertImageWithBands.set({
            "system:time_start": start_date.millis(),
            "system:time_end": end_date.millis(),
            "weekNumber": weekNumber,
            "year": year
        })

        # Create the list of names for the bands
        bandNames = ["combinedAlerts", "glad", "sar"]

        # Rename the bands in the image
        alertImageWithBands = alertImageWithBands.rename(bandNames)

        # Create the name for the image
        imageName = "khm_def_alert_img_{}_{}".format(year, f"{weekNumber:02d}")

        # Define the export task
        exportTask = ee.batch.Export.image.toAsset(
            image=alertImageWithBands,
            description='Export_to_asset',
            assetId=collection_id + imageName,
            scale=10,  # Adjust this based on the native resolution of your image
            region=aoi.geometry(),
            maxPixels=1e13
        )

        # Start the export task
        exportTask.start()

        # Monitor the task
        print(exportTask.status())

        return exportTask.status()

    alert_date_band = f"alertdate{str(year)[-2:]}"
    alert_date_band2 = f"alertDate{str(year)[-2:]}"
    filter_start_date = f"{year}-01-01"
    filter_end_date = f"{year}-12-31"
    # Define the Image Collection ID.
    collection_id = 'projects/cemis-camp/assets/khForestAlert/combinedDeforestationAlertv2/'

    aoiKhADM0 = ee.FeatureCollection("projects/servir-mekong/admin/KHM_adm0")

    sarAlert = ee.ImageCollection("projects/cemis-camp/assets/khForestAlert/sarBiWeeklyAlert").select(
        alert_date_band).filterDate(filter_start_date, filter_end_date).min()
    gladAlert = ee.ImageCollection(
        'projects/glad/alert/UpdResult').select(alert_date_band2).max()

    process_biweekly_period(year, biweekly_period, sarAlert,
                            gladAlert, aoiKhADM0, collection_id)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combineDeforestationAlert.py <year> <biweekly_period>")
        sys.exit(1)

    year = int(sys.argv[1])
    biweekly_period = sys.argv[2]
    main(year, biweekly_period)
