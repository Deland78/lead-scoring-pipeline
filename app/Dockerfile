# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures we don't store the download cache, making the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
# Add timeout and worker configuration
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
