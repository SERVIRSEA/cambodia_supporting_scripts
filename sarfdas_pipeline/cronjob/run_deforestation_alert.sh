#!/bin/bash

# Calculate the current week of the year
current_week=$(date +%V)
current_year=$(date +%Y)

# Calculate the target week (4 weeks before)
target_week=$((current_week - 4))

# If the target week is less than 1, adjust for previous year
if [ $target_week -lt 1 ]; then
  target_week=$((52 + target_week))
  current_year=$((current_year - 1))
fi

# Run only for odd weeks
if ((target_week % 2 == 1)); then
  echo "Running update deforestation alert for year $current_year, week $target_week"
  # Run the Python script with the calculated year and week
  python3 ~/forestAlert/sarAlerts/updatedeforestationalert.py $current_year $target_week
else
  echo "Week $target_week is not an odd week. Skipping execution."
fi