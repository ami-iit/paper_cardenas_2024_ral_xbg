FROM pytorch/pytorch

RUN pip install pipenv
RUN apt-get update && apt-get install nano ffmpeg libsm6 libxext6  -y

ENV OMP_NUM_THREADS=16
WORKDIR /element
COPY ["Pipfile", "Pipfile.lock", "./"]
COPY "src/preprocessing/" "src/preprocessing/"
COPY "src/XBG/" "src/XBG/"

RUN pipenv install --system --deploy
WORKDIR src/XBG
