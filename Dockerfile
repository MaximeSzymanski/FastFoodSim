FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Updated apt-get for Debian Trixie (Debian 13)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libgl1 \
    libglx-mesa0 \
    swig \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the project and dependencies
COPY pyproject.toml .
# Create a dummy src so pip can install the project structure
RUN mkdir src && pip install --no-cache-dir -e .

# Copy the actual source code (which brings in src/rl/train_all.py etc.)
COPY . .

# Final link of the editable package
RUN pip install --no-cache-dir -e .

# OPTION 2 FIX: Tell Docker to run the file as a module from inside src.rl
CMD ["python", "-m", "src.rl.train_all"]
