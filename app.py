from flask import Flask, render_template, request, jsonify
from music_genre_classifier import predict_genre
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='Generated Directories/templates')


@app.route('/')
def index():
    """Render the main page with the classification form."""
    # Check if both models exist
    mendeley_model_exists = os.path.exists("Generated Directories/mendeley_model")
    merged_model_exists = os.path.exists("Generated Directories/trained_model")

    return render_template('index.html',
                           mendeley_model_exists=mendeley_model_exists,
                           merged_model_exists=merged_model_exists)


@app.route('/classify', methods=['POST'])
def classify():
    """Handle the classification request and return the results as JSON."""
    lyrics = request.form.get('lyrics', '')
    model_type = request.form.get('model_type', 'merged')  # Default to merged (8 genres)

    if not lyrics:
        logger.warning("Classification request received with empty lyrics")
        return jsonify({"error": "No lyrics provided"}), 400

    logger.info(f"Classification request received with {len(lyrics)} characters, model type: {model_type}")

    # Determine which model to use
    model_path = "Generated Directories/mendeley_model" if model_type == "mendeley" else "Generated Directories/trained_model"

    # Get genre predictions
    try:
        genre_probabilities = predict_genre(lyrics, model_path)

        if "error" in genre_probabilities:
            logger.error(f"Error during prediction: {genre_probabilities['error']}")
            return jsonify(genre_probabilities), 500

        logger.info(f"Classification successful, found {len(genre_probabilities)} genres")
        return jsonify(genre_probabilities)
    except Exception as e:
        logger.error(f"Unexpected error during classification: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Ensure the trained model exists
    if not os.path.exists("Generated Directories/trained_model") and not os.path.exists(
            "Generated Directories/mendeley_model"):
        logger.error("No models found. Please run music_genre_classifier.py first")
        print("Error: No models found. Please run music_genre_classifier.py first")
        exit(1)

    # Create templates directory if it doesn't exist
    if not os.path.exists("Generated Directories/templates"):
        os.makedirs("Generated Directories/templates")

    # Create index.html if it doesn't exist
    if not os.path.exists("Generated Directories/templates/index.html"):
        logger.info("Creating index.html template")
        with open("Generated Directories/templates/index.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Music Genre Classifier</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .model-selector {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .model-selector p {
            margin-top: 0;
            font-weight: bold;
        }
        .model-selector label {
            display: block;
            margin: 5px 0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 200px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: block;
            margin: 0 auto;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
        }
        .chart-container {
            position: relative;
            height: 400px;
            width: 100%;
        }
        .loading {
            text-align: center;
            margin-top: 20px;
            display: none;
        }
        .predicted-genre {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Music Genre Classifier</h1>
        <div class="model-selector">
            <p>Select model:</p>
            <label>
                <input type="radio" name="model_type" value="mendeley" {% if mendeley_model_exists %}checked{% else %}disabled{% endif %}>
                7 Genres (Mendeley Dataset)
            </label>
            <label>
                <input type="radio" name="model_type" value="merged" {% if merged_model_exists %}checked{% else %}disabled{% endif %}>
                8 Genres (Merged Dataset)
            </label>
        </div>
        <textarea id="lyrics" placeholder="Paste song lyrics here..."></textarea>
        <button onclick="classifyLyrics()">Classify</button>
        <div class="loading" id="loading">Analyzing lyrics...</div>
        <div id="result">
            <div class="predicted-genre" id="predicted-genre"></div>
            <div class="chart-container">
                <canvas id="genreChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let chart = null;

        function classifyLyrics() {
            const lyrics = document.getElementById('lyrics').value;
            if (!lyrics) {
                alert('Please enter lyrics');
                return;
            }

            // Get selected model type
            const modelTypeElements = document.getElementsByName('model_type');
            let modelType = 'merged'; // Default
            for (const radioBtn of modelTypeElements) {
                if (radioBtn.checked) {
                    modelType = radioBtn.value;
                    break;
                }
            }

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            if (chart) {
                chart.destroy();
                chart = null;
            }
            document.getElementById('predicted-genre').textContent = '';

            fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `lyrics=${encodeURIComponent(lyrics)}&model_type=${encodeURIComponent(modelType)}`
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data, modelType);
            })
            .catch(error => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';

                console.error('Error:', error);
                alert('An error occurred during classification');
            });
        }

        function displayResults(data, modelType) {
            const genres = Object.keys(data);
            const probabilities = Object.values(data);

            // Find the genre with the highest probability
            let maxProb = 0;
            let predictedGenre = '';
            genres.forEach((genre, index) => {
                if (probabilities[index] > maxProb) {
                    maxProb = probabilities[index];
                    predictedGenre = genre;
                }
            });

            // Display the predicted genre
            document.getElementById('predicted-genre').textContent = 
                `Predicted Genre: ${predictedGenre} (${(maxProb * 100).toFixed(2)}%)`;

            // Sort data by probability
            const sortedIndices = probabilities.map((prob, index) => index)
                .sort((a, b) => probabilities[b] - probabilities[a]);

            const sortedGenres = sortedIndices.map(i => genres[i]);
            const sortedProbabilities = sortedIndices.map(i => probabilities[i]);

            // Add model type to the chart title
            const modelName = modelType === 'mendeley' ? 'Mendeley Dataset (7 Genres)' : 'Merged Dataset (8 Genres)';

            // Create chart
            const ctx = document.getElementById('genreChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sortedGenres,
                    datasets: [{
                        label: 'Probability',
                        data: sortedProbabilities.map(p => p * 100), // Convert to percentage
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.7)',
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 206, 86, 0.7)',
                            'rgba(75, 192, 192, 0.7)',
                            'rgba(153, 102, 255, 0.7)',
                            'rgba(255, 159, 64, 0.7)',
                            'rgba(199, 199, 199, 0.7)',
                            'rgba(83, 102, 255, 0.7)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)',
                            'rgba(199, 199, 199, 1)',
                            'rgba(83, 102, 255, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `Genre Classification Results - ${modelName}`,
                            font: {
                                size: 18
                            }
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.raw.toFixed(2)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Probability (%)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>""")

    logger.info("Starting web application server")
    print("Starting Music Genre Classifier web application on http://localhost:8080")
    app.run(debug=False, host='0.0.0.0', port=8080)