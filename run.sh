#!/bin/bash
# run.sh - Script to run the Music Genre Classifier web interface

echo "╔══════════════════════════════════════════════════════════╗"
echo "║             Music Genre Classifier Launcher               ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Check for required packages
echo "Checking and installing required packages..."
PACKAGES=("flask" "pyspark" "findspark")

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

# Create templates directory if it doesn't exist
mkdir -p "templates"

# Verify that the model exists
if [ ! -d "Trained_Final_Model" ]; then
    echo "Error: Model not found in Trained_Final_Model directory."
    echo "Please ensure the pre-trained model is available in the Trained_Final_Model directory."
    exit 1
fi
echo "Pre-trained model found in Trained_Final_Model directory."

# Start the web application and open the browser
echo "Starting web application..."
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