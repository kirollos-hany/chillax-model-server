FROM python:3.8-slim-buster

COPY Models /app/Models
COPY chillax_model_server.py /app
COPY chillax_models.py /app
COPY python-libraries.sh /app

WORKDIR /app
RUN chmod +x ./python-libraries.sh
RUN /bin/bash ./python-libraries.sh
CMD [ "python3", "chillax_model_server.py" ] 