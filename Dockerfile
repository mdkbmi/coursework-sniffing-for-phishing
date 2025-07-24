FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jre \
    curl \
    git \
    unzip \
    make \
    g++ \
    && apt-get clean

# Create working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/predict_email.py ./src/


# Begin LanguageTool + fastText Setup
# Create dependencies directory
RUN mkdir -p /dependencies

# Download and unzip LanguageTool snapshot
RUN curl -L -o /dependencies/LanguageTool.zip https://internal1.languagetool.org/snapshots/LanguageTool-latest-snapshot.zip \
    && unzip /dependencies/LanguageTool.zip -d /dependencies \
    && rm /dependencies/LanguageTool.zip

# Clone and build fastText
RUN git clone https://github.com/facebookresearch/fastText.git /dependencies/fastText \
    && cd /dependencies/fastText \
    && make

# Download lid.176.bin model to fastText folder
RUN curl -Lo /dependencies/fastText/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Create server.properties for LanguageTool
RUN LT_DIR=$(find /dependencies -maxdepth 1 -type d -name 'LanguageTool-*' | head -n 1) && \
    echo "fasttextModel=/dependencies/fastText/lid.176.bin" >> "$LT_DIR/server.properties" && \
    echo "fasttextBinary=/dependencies/fastText/fasttext" >> "$LT_DIR/server.properties"
# End LanguageTool + fastText Setup

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "src/app.py"]
