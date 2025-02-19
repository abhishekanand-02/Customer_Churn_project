# Use the official Python 3.8 slim-buster image as the base image  
FROM python:3.8-slim-buster  

# Step 1: Update the apt package list and install awscli  
RUN apt update -y && apt install -y awscli  

# Step 2: Set the working directory inside the container to /app  
WORKDIR /app  

# Step 3: Copy the contents of the current directory (including your model, scripts, etc.) into the /app directory in the container  
COPY . /app  

# Step 4: Install the Python dependencies listed in requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt  


EXPOSE 8501


# Step 5: Set the command to run Streamlit when the container starts 
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CMD ["streamlit", "run", "app.py"]
# CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]


