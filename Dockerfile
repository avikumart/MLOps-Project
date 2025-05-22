FROM python:3.11-slim

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set the working directory
WORKDIR /Yellow-Taxi-modeling-and-deployment

RUN pip install -U pip
RUN pip install -U pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy 

COPY . .

# Expose the port the app runs one
EXPOSE 6000

ENTRYPOINT [ "uvicorn", "Yellow-Taxi-modeling-and-deployment.main:app", "--host", "0.0.0.0", "--port", "6000" ]