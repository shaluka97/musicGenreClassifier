#!/bin/bash

echo "╔══════════════════════════════════════════════════════════╗"
echo "║             Music Genre Classifier Launcher               ║"
echo "╚══════════════════════════════════════════════════════════╝"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3."
    exit 1
fi

echo "Checking and installing required packages..."
PACKAGES=("flask" "pyspark" "findspark" "numpy")

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

mkdir -p "templates"

if [ ! -d "Trained_Final_Model" ]; then
    echo "Error: Model not found in Trained_Final_Model directory."
    echo "Please ensure the pre-trained model is available in the Trained_Final_Model directory."
    exit 1
fi
echo "Pre-trained model found in Trained_Final_Model directory."

echo "Starting web application..."
echo "The web interface will be available at http://localhost:8080"
echo "Automatically opening web browser..."

open_browser() {
    local url="http://localhost:8080"

    sleep 2

    case "$OSTYPE" in
        darwin*)
            open "$url"
            ;;
        msys*|cygwin*|mingw*)
            start "$url"
            ;;
        linux*)
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
        *)
            echo "Unsupported OS for automatic browser opening. Please navigate to $url manually."
            ;;
    esac
}

python3 app.py &
APP_PID=$!

open_browser

echo "Press Ctrl+C to stop the application when you're done."

trap "kill $APP_PID 2>/dev/null" INT TERM

wait $APP_PID