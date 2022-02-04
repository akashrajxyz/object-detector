FROM ubuntu
ENV DEBIAN_FRONTEND noninteractive
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y  install python3.9 python3-pip
RUN python3.9 -m pip install setuptools && python3.9 -m pip install -r requirements.txt
RUN python3.9 -m pip  install tf-nightly
EXPOSE 5000
ENTRYPOINT [ "python3.9" ]
CMD [ "api.py" ]
