FROM python:3.10.9
COPY . /app
WORKDIR /app
RUN conda install -c apple tensorflow-deps && pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app