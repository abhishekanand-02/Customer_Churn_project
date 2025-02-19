# Use the official Python 3.8 slim-buster image as the base image
# If you want Python 3.12.8, change it to FROM python:3.12.8-slim-buster
FROM python:3.8-slim-buster

# Step 1: Update the apt package list and install awscli
# The `-y` flag automatically answers 'yes' to prompts during installation
RUN apt update -y && apt install -y awscli

# Step 2: Set the working directory inside the container to /app
WORKDIR /app

# Step 3: Copy the contents of the current directory (including your model, scripts, etc.) into the /app directory in the container
COPY . /app

# Step 4: Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Set the command to run your Python application when the container starts
CMD ["python3", "app.py"]