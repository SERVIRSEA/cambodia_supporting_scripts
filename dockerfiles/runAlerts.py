import ee
import os
import sys
from datetime import date, timedelta
# import ee.mapclient


def main(year, week):
    def get_biweekly_dates(year, week):
        # Calculate the start week
        start_week = week if week % 2 == 1 else week - 1
        end_week = start_week + 1

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

        # Return dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        return start_date_str, end_date_str

    ee.Initialize()

    mpl = ee.FeatureCollection(
        "projects/servir-mekong/admin/KHM_adm0").geometry().bounds()

    # mpl = ee.FeatureCollection("projects/cemis-camp/assets/MorodokBaitang/UMB_Project_site_buffer").filter(ee.Filter.eq("NAME","Central Phnom Kravanh"))
    # mpl = ee.FeatureCollection("projects/cemis-camp/assets/wdpa/WDPA_WDOECM_KHM_shp-polygons").filter(ee.Filter.eq("NAME","Prey Lang"))
    MODE = 'DESCENDING'

    region = mpl  # .merge(protected).geometry()

    # Get bi-weekly start and end dates
    start, end = get_biweekly_dates(year, week)\
        # end = ee.Date(cdate.strftime("%Y-%m-%d"))

    s1Collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, end)\
        .filterBounds(region)\
        .sort("system:time_start")\
        .filter(ee.Filter.eq('orbitProperties_pass', MODE))\
        .sort("system:time_start", True)\
        .aggregate_histogram("system:index").getInfo()

    counter = 0
    for item in s1Collection:
        print(item)
        if os.path.exists(r"/home/ubuntu/forestAlert/data/" + item + "/" + item + ".tif"):
            print(item, "already exists!")
        else:
            # print(item)
            os.system("docker run -v /home/ubuntu/forestAlert/data:/root/data -e IMAGE_ID=" +
                      item + " alertscambodia")

        counter += 1
    print(f"Running runAlerts for year {year} and week {week}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python runAlerts.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    main(year, week)
