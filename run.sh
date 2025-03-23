#!/bin/bash
# run.sh - Script to run the Music Genre Classifier

# Set Java security options for macOS compatibility
export JAVA_OPTS="-Djava.security.manager=allow"
export PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraJavaOptions=-Djava.security.manager=allow pyspark-shell"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║             Music Genre Classifier Launcher               ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Check for PySpark and install required packages
python3 -c "import pyspark" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PySpark is not installed. Installing it now..."
    pip3 install pyspark==3.1.2
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PySpark. Please install it manually using:"
        echo "pip install pyspark==3.1.2"
        exit 1
    fi
fi

# Install findspark to help locate Spark installation
python3 -c "import findspark" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing findspark to help with Spark initialization..."
    pip3 install findspark
fi

# Check for required packages
echo "Checking and installing required packages..."
PACKAGES=("flask" "pandas" "numpy" "matplotlib")

for package in "${PACKAGES[@]}"; do
    python3 -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing $package..."
        pip3 install $package
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to install $package. Some functionality may be limited."
        fi
    fi
done

# Create required directories if they don't exist
mkdir -p "Data Files"
mkdir -p "Generated Directories"
mkdir -p "Generated Directories/templates"

# Check for required files
if [ ! -f "Data Files/Student_dataset.csv" ]; then
    # Check if it's in the root directory and move it
    if [ -f "Student_dataset.csv" ]; then
        echo "Moving Student_dataset.csv to Data Files directory..."
        mv Student_dataset.csv "Data Files/"
    else
        echo "Error: Student_dataset.csv not found. Please place it in the Data Files directory."
        exit 1
    fi
fi

if [ ! -f "Data Files/Mendeley_dataset.csv" ]; then
    # Check if it's in the root directory and move it
    if [ -f "Mendeley_dataset.csv" ]; then
        echo "Moving Mendeley_dataset.csv to Data Files directory..."
        mv Mendeley_dataset.csv "Data Files/"
    else
        # Check for common alternative names
        if [ -f "mendeley_dataset.csv" ]; then
            echo "Found mendeley_dataset.csv, using it instead."
            mv mendeley_dataset.csv "Data Files/Mendeley_dataset.csv"
        else
            echo "Error: Could not find the Mendeley dataset. Please place it in the Data Files directory."
            exit 1
        fi
    fi
fi

# Create directories for the application
mkdir -p "Generated Directories/templates"

# Step 1: Run the music_genre_classifier.py to merge datasets and train the models if needed
echo "Step 1: Preparing data and models..."

# Check if we're running on macOS (which has the security manager issue)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS environment, using specific Java security settings..."
    JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$JAVA_HOME" ]; then
        echo "Found Java at: $JAVA_HOME"
        export JAVA_HOME
    fi
fi

python3 music_genre_classifier.py

# Check if the previous step was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to prepare data and models. Check the error messages above."
    exit 1
fi

# Check if both models are available
if [ ! -d "Generated Directories/mendeley_model" ]; then
    echo "Warning: Mendeley model (7 genres) is not available."
fi

if [ ! -d "Generated Directories/trained_model" ]; then
    echo "Warning: Merged model (8 genres) is not available."
fi

if [ ! -d "Generated Directories/mendeley_model" ] && [ ! -d "Generated Directories/trained_model" ]; then
    echo "Error: No models are available. Both training steps failed."
    exit 1
fi

# Step 2: Start the web application and open the browser
echo "Step 2: Starting web application..."
echo "The web interface will be available at http://localhost:8080"
echo "Automatically opening web browser..."

# Function to open browser based on OS
open_browser() {
    local url="http://localhost:8080"

    # Give the server a moment to start
    sleep 2

    # Detect the OS and use the appropriate command to open the browser
    case "$OSTYPE" in
        darwin*)  # macOS
            open "$url"
            ;;
        msys*|cygwin*|mingw*)  # Windows
            start "$url"
            ;;
        linux*)  # Linux
            if command -v xdg-open &> /dev/null; then
                xdg-open "$url"
            elif command -v gnome-open &> /dev/null; then
                gnome-open "$url"
            elif command -v kde-open &> /dev/null; then
                kde-open "$url"
            else
                echo "Could not automatically open browser. Please navigate to $url manually."
            fi
            ;;
        *)  # Other OS
            echo "Unsupported OS for automatic browser opening. Please navigate to $url manually."
            ;;
    esac
}

# Start the web app in the background and open the browser
python3 app.py &
APP_PID=$!

# Open the browser
open_browser

echo "Press Ctrl+C to stop the application when you're done."

# Trap Ctrl+C to gracefully shut down
trap "kill $APP_PID 2>/dev/null" INT TERM

# Wait for the Flask app to finish
wait $APP_PID