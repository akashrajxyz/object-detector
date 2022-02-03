# Object Detection
An API written in flask which takes image file as an POST request and
return the number of objects got detected in image.

## Requirements
- Python 3.7.x
- Tensorflow 2.x
- OpenCV 4.x
- Flask 2.x 

## Installation

```sh
git clone https://github.com/dasagreeva/object-detector.git
cd object-detector
```

### Running with Gunicorn
```
pip install -r requirements.txt
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 api:app

```

### Running with docker
```
git clone https://github.com/dasagreeva/object-detector.git
cd object-detector
docker image build -t ml-app:1.0 .
docker run -p 5000:5000 -d ml-app
```

## Usage
After running the server on some port (it use port =5000= by default).
Send a POST request using =curl=.
```sh
curl -F"image=@test_image_file.jpg" https://localhost:5000/1/predict
{"count": 3}
```
