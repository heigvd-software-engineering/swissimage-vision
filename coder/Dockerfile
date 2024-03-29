FROM codercom/enterprise-base:ubuntu

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN rm requirements.txt

