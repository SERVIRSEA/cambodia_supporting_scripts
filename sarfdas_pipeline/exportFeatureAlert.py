import ee
from datetime import date, timedelta
import sys


def main(year, biweekly_period):
    # Your existing code
    print(
        f"Running exportFeatureAlert for year {year} and biweekly period {biweekly_period}")

    # Authenticate to the Earth Engine servers
    ee.Initialize()

    biweekly_period = int(biweekly_period)

    def get_biweekly_dates(year, biweekly_period):
        # Calculate the start week
        start_week = biweekly_period

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

    # Get the start and end dates for the bi-weekly period
    start_date, end_date = get_biweekly_dates(year, biweekly_period)

    # Define the ImageCollection.
    ic = ee.ImageCollection(
        "projects/cemis-camp/assets/khForestAlert/combinedDeforestationAlertv2")

    # Filter the ImageCollection to cover only the year 2023.
    ic_filtered = ic.filterDate(start_date, end_date)

    # Function to process the ImageCollection.

    def process_collection(image):
        start_date = ee.Date(image.get('system:time_start'))
        # Assuming the image has this property
        end_date = ee.Date(image.get('system:time_end'))

        # Function to convert day of year to date string
        def convert_doy_to_date(doy, year):
            date = ee.Date.fromYMD(year, 1, 1).advance(
                ee.Number(doy).subtract(1), 'day')
            return date.format('YYYY-MM-dd')

        # Get the year from the start_date
        year = ee.Number(start_date.get('year'))

        # Select the 'combinedAlerts' band and threshold it to create a binary image.
        binary = image.select('combinedAlerts').eq(1)

        # Convert the binary image to vectors (polygons).
        vectors = binary.reduceToVectors(
            geometryType='polygon',
            reducer=ee.Reducer.countEvery(),
            scale=10,  # Adjust scale if necessary.
            maxPixels=1e13
        )

        # Filter out polygons with area less than 50 pixels (assuming 30m resolution, adjust if different).
        vectors = vectors.filter(ee.Filter.gt('count', 50))

        # Function to calculate the max value of 'glad', 'sar', and 'tropico' bands.
        def calculate_max_values(feature):
            feature = ee.Feature(feature)
            min_glad = image.select('glad').reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=feature.geometry(),
                scale=10,  # Adjust scale if necessary.
                maxPixels=1e13
            ).get('glad')

            min_sar = image.select('sar').reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=feature.geometry(),
                scale=10,  # Adjust scale if necessary.
                maxPixels=1e13
            ).get('sar')

            # Ensure min_glad and min_sar are valid numbers
            min_glad = ee.Number(min_glad)
            min_sar = ee.Number(min_sar)

            # Check if min_glad and min_sar are valid before converting to dates
            min_glad_date = ee.Algorithms.If(
                min_glad,
                convert_doy_to_date(min_glad, year),
                None
            )

            min_sar_date = ee.Algorithms.If(
                min_sar,
                convert_doy_to_date(min_sar, year),
                None
            )

            # Format the start and end dates as strings
            start_date_str = start_date.format('YYYY-MM-dd')
            end_date_str = end_date.format('YYYY-MM-dd')

            # Set the max values as properties.
            return feature.set({
                'minGlad': min_glad,
                'minSar': min_sar,
                'minGladDate': min_glad_date,
                'minSarDate': min_sar_date,
                'startDate': start_date_str,
                'endDate': end_date_str
            })

        # Map over each feature to calculate the max values.
        vectors = vectors.map(calculate_max_values)

        # Create a 50m buffer for each feature.
        # Buffer by 50 meters.
        buffered = vectors.map(lambda feature: feature.buffer(50))

        # Create a -25m buffer for each feature.
        # Inner buffer by -25 meters.
        inner_buffered = buffered.map(lambda feature: feature.buffer(-25))

        # Return the final features.
        return inner_buffered

    # Apply the function to each image in the collection.
    processed_collection = ic_filtered.map(process_collection)

    # Flatten the collection of collections into a single FeatureCollection.
    final_feature_collection = processed_collection.flatten()

    # Export the FeatureCollection.
    task = ee.batch.Export.table.toAsset(
        collection=final_feature_collection,
        description='export_feature_deforestation_alert',
        assetId=f'projects/cemis-camp/assets/khForestAlert/sarfdas_feature_alert/khm_def_sar_v2_{year}_{biweekly_period:02d}'
    )

    # Start the export task.
    task.start()

    # Monitor the task.
    print(task.status())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python exportFeatureAlert.py <year> <biweekly_period>")
        sys.exit(1)

    year = int(sys.argv[1])
    biweekly_period = sys.argv[2]
    main(year, biweekly_period)
