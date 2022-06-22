FROM ubuntu:22.04

RUN apt update && apt -y install tzdata
ENV TZ=Asia/Tokyo

RUN apt update && apt -y install vim python3 python3-pip python3-tk libgl1-mesa-dev libgtk2.0-dev && \
    python3 -m pip install numpy matplotlib opencv-python opencv-contrib-python tqdm
