FROM python:3.9

WORKDIR /app

COPY /cell_images/ /app/cell_images/
COPY /csv_files/ /app/csv_files/
COPY /pages/ /app/pages/
COPY /samples/ /app/samples/
COPY premium-odyssey-378518-934cec99d0b6.json /app/
COPY requirements.txt /app/
COPY streamlit_download_button.py /app/
COPY Table_Extractor_App.py /app/
COPY TableExtraction.py /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Table_Extractor_App.py", "--server.port=8501", "--server.address=0.0.0.0"]