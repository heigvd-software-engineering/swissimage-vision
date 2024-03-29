FROM pytorch/pytorch:1.7.1

WORKDIR /app

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
