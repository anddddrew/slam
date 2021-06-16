FROM 3.8-alpine

WORKDIR /

COPY requirements.txt /

RUN pip3 install -r requirements.txt

RUN python3 slam.py basevideo.mp4

