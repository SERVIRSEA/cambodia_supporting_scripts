FROM python:3.8-slim-buster

# Define environment variable
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Update and install dependencies
RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libproj-dev \
    libgdal-dev \
    gdal-bin \
    python3-gdal

RUN apt-get update && apt-get install -y curl gnupg && \
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
        apt-get update && apt-get install -y google-cloud-sdk

# Install other Python packages
RUN pip install rasterio fiona shapely geopandas tensorflow
RUN pip install earthengine-api
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
RUN pip install retrying
RUN pip install rio-cogeo
RUN pip install tqdm

RUN mkdir /root/data
VOLUME /root/data

# Copy code
COPY main.py /root/main.py
COPY model.py /root/model.py
COPY CambodiaMekongv3.h5 /root/model.h5
COPY credentials /root/.config/earthengine/credentials

# Run the command
CMD ["python", "/root/main.py"]
