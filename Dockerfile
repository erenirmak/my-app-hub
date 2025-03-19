# Use a lightweight Python base image with the version you need
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy Python dependencies file and install requirements
COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your Flask application code into container
COPY . /app

# Expose Flask application's port (e.g., 5000)
EXPOSE 5000

# Specify Flask app environment variable
ENV FLASK_APP=app.py

# Command to run Flask application
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]