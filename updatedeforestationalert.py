import subprocess
import sys
import logging
from datetime import datetime
import time


def run_script(script_path, args):
    """Run a python script with the given arguments and wait for it to finish."""
    result = subprocess.run(
        [sys.executable, script_path] + args, capture_output=True, text=True)
    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise Exception(
            f"Script {script_path} failed with exit code {result.returncode}")


def calculate_biweekly_period(week):
    """Calculate the start and end weeks for a given bi-weekly period."""
    start_week = week if week % 2 == 1 else week - 1
    end_week = start_week + 1
    return start_week, end_week


def main(year, week):
    # Calculate bi-weekly period
    start_week, end_week = calculate_biweekly_period(week)
    biweekly_period = f"{start_week}-{end_week}"
    logging.info(
        f"Processing for Year: {year}, Bi-weekly Period: {biweekly_period}")

    # Define script paths
    run_alerts_path = '/home/ubuntu/forestAlert/sarAlerts/dockerfiles/runAlerts.py'
    upload_files_path = '/home/ubuntu/forestAlert/sarAlerts/sarfdas_pipeline/uploadFiles.py'
    upload_gee_path = '/home/ubuntu/forestAlert/sarAlerts/sarfdas_pipeline/uploadGEE.py'
    sar_alert_biweekly_path = '/home/ubuntu/forestAlert/sarAlerts/sarfdas_pipeline/sarAlertBiWeekly.py'
    combine_deforestation_alert_path = '/home/ubuntu/forestAlert/sarAlerts/sarfdas_pipeline/combineDeforestationAlert.py'
    export_feature_alert_path = '/home/ubuntu/forestAlert/sarAlerts/sarfdas_pipeline/exportFeatureAlert.py'

    try:
        # Run runAlerts.py
        logging.info("Running runAlerts.py...")
        run_script(run_alerts_path, [str(year), str(week)])

        # Run uploadFiles.py
        logging.info("Running uploadFiles.py...")
        run_script(upload_files_path, [])

        # Run uploadGEE.py
        logging.info("Running uploadGEE.py...")
        run_script(upload_gee_path, [])

        # Introduce a delay of 1 hour (3600 seconds)
        logging.info(
            "Waiting for 1 hour before running the next scripts to make sure gee tasks finisih...")
        time.sleep(3600)

        # Run sarAlertBiWeekly.py
        logging.info("Running sarAlertBiWeekly.py...")
        run_script(sar_alert_biweekly_path, [str(year), str(week)])

        # Introduce a delay of 1 hour (3600 seconds)
        logging.info(
            "Waiting for 1 hour before running the next scripts to make sure gee tasks finisih...")
        time.sleep(3600)

        # Run combineDeforestationAlert.py
        logging.info("Running combineDeforestationAlert.py...")
        run_script(combine_deforestation_alert_path, [str(year), str(week)])

        # Introduce a delay of 1 hour (3600 seconds)
        logging.info(
            "Waiting for 1 hour before running the next scripts to make sure gee tasks finisih...")
        time.sleep(3600)

        # Run exportFeatureAlert.py
        logging.info("Running exportFeatureAlert.py...")
        run_script(export_feature_alert_path, [str(year), str(week)])

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python updatedeforestationalert.py <year> <week>")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])

    # Set up logging
    logging.basicConfig(filename=f'/home/ubuntu/forestAlert/sarAlerts/log/update_deforestation_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting update deforestation alert process")

    try:
        main(year, week)
        logging.info(
            "Update deforestation alert process completed successfully")
    except Exception as e:
        logging.error(f"Update deforestation alert process failed: {e}")
        sys.exit(1)
