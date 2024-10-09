# Base image with Python
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory (the app) into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the app will run (default Streamlit port)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "frontend.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.port=8501", "--server.headless=true"]
