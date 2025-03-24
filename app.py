from flask import Flask, render_template, request, jsonify
from music_genre_classifier import predict_genre
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    model_exists = os.path.exists("Trained_Final_Model")
    return render_template('index.html', model_exists=model_exists)


@app.route('/classify', methods=['POST'])
def classify():
    lyrics = request.form.get('lyrics', '')

    if not lyrics:
        logger.warning("Classification request received with empty lyrics")
        return jsonify({"error": "No lyrics provided"}), 400

    logger.info(f"Classification request received with {len(lyrics)} characters")

    try:
        genre_probabilities = predict_genre(lyrics, "Trained_Final_Model")

        if "error" in genre_probabilities:
            logger.error(f"Error during prediction: {genre_probabilities['error']}")
            return jsonify(genre_probabilities), 500

        logger.info(f"Classification successful, found {len(genre_probabilities)} genres")
        return jsonify(genre_probabilities)
    except Exception as e:
        logger.error(f"Unexpected error during classification: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists("Trained_Final_Model"):
        logger.error("No model found. Please run music_genre_classifier.py first")
        print("Error: No model found. Please run music_genre_classifier.py first")
        exit(1)

    if not os.path.exists("templates"):
        os.makedirs("templates")

    if not os.path.exists("templates/index.html"):
        logger.info("Creating index.html template")
        with open("templates/index.html", "w") as f:
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
        <textarea id="lyrics" placeholder="Paste song lyrics here..."></textarea>
        <button onclick="classifyLyrics()" {% if not model_exists %}disabled{% endif %}>Classify</button>
        {% if not model_exists %}
        <p style="text-align: center; color: red;">Model not found. Please run music_genre_classifier.py first.</p>
        {% endif %}
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
                body: `lyrics=${encodeURIComponent(lyrics)}`
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred during classification');
            });
        }

        function displayResults(data) {
            const genres = Object.keys(data);
            const probabilities = Object.values(data);

            let maxProb = 0;
            let predictedGenre = '';
            genres.forEach((genre, index) => {
                if (probabilities[index] > maxProb) {
                    maxProb = probabilities[index];
                    predictedGenre = genre;
                }
            });

            document.getElementById('predicted-genre').textContent = 
                `Predicted Genre: ${predictedGenre} (${(maxProb * 100).toFixed(2)}%)`;

            const sortedIndices = probabilities.map((prob, index) => index)
                .sort((a, b) => probabilities[b] - probabilities[a]);

            const sortedGenres = sortedIndices.map(i => genres[i]);
            const sortedProbabilities = sortedIndices.map(i => probabilities[i]);

            const ctx = document.getElementById('genreChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sortedGenres,
                    datasets: [{
                        label: 'Probability',
                        data: sortedProbabilities.map(p => p * 100),
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
                            text: 'Genre Classification Results',
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