import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lower, col

# Set environment variables for Java security on macOS
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'
os.environ['JAVA_OPTS'] = '-Djava.security.manager=allow'

# Try to import findspark if available (helps locate Spark installation)
try:
    import findspark

    findspark.init()

except ImportError:
    print("Note: findspark not found. This is fine if your PySpark setup is already configured.")

def create_pipeline():
    """Creates and returns the ML pipeline for text classification."""
    tokenizer = RegexTokenizer(inputCol="lyrics", outputCol="words", pattern="\\W+")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    indexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="keep")
    classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    return Pipeline(stages=[tokenizer, remover, hashing_tf, idf, indexer, classifier])

def train_mendeley_model():
    """
    Trains a music genre classification model using ONLY the Mendeley dataset (7 genres).
    Saves the trained model to disk.
    """
    print("\n=== STAGE 1: TRAINING MODEL ON MENDELEY DATASET (7 GENRES) ===\n")
    print("Initializing Spark session...")
    # Initialize Spark session with simplified configuration for macOS compatibility
    try:
        spark = SparkSession.builder \
            .appName("Music Genre Classification - Mendeley") \
            .master("local[*]") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("\nTrying alternative Spark initialization...")

        # Fallback to minimal configuration
        spark = SparkSession.builder \
            .appName("Music Genre Classification") \
            .master("local[*]") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

    # Load the Mendeley dataset
    print("Loading Mendeley dataset...")
    try:
        mendeley_data = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("Data Files/Mendeley_dataset.csv")

        # Verify that we have the expected columns
        required_columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
        for column in required_columns:
            if column not in mendeley_data.columns:
                raise Exception(f"Missing required column: {column}")

        # Check for the number of unique genres
        num_genres = mendeley_data.select("genre").distinct().count()
        print(f"Number of unique genres in the Mendeley dataset: {num_genres}")
        genres = [row.genre for row in mendeley_data.select("genre").distinct().collect()]
        print(f"Genres in the Mendeley dataset: {', '.join(genres)}")

    except Exception as e:
        print(f"Error loading Mendeley dataset: {e}")
        return None

    # Prepare the data
    print("Preprocessing data...")
    clean_data = mendeley_data.withColumn("lyrics", lower(col("lyrics")))

    # Split the data into training and test sets
    training_data, test_data = clean_data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training data size: {training_data.count()}, Test data size: {test_data.count()}")

    # Create the ML pipeline
    print("Building ML pipeline...")
    pipeline = create_pipeline()

    # Train the model
    print("Training model on Mendeley dataset... (this may take a few minutes)")
    model = pipeline.fit(training_data)

    # Evaluate the model
    print("Evaluating model performance...")
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test accuracy (Mendeley model): {accuracy}")

    # Save the model
    print("Saving Mendeley model...")
    model.write().overwrite().save("Generated Directories/mendeley_model")
    print("Mendeley model training complete!")

    return model

def train_merged_model():
    """
    Trains a music genre classification model using the merged dataset (8 genres).
    Saves the trained model to disk.
    """
    print("\n=== STAGE 2: TRAINING MODEL ON MERGED DATASET (8 GENRES) ===\n")
    print("Initializing Spark session...")
    # Initialize Spark session with simplified configuration for macOS compatibility
    try:
        spark = SparkSession.builder \
            .appName("Music Genre Classification - Merged") \
            .master("local[*]") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("\nTrying alternative Spark initialization...")

        # Fallback to minimal configuration
        spark = SparkSession.builder \
            .appName("Music Genre Classification") \
            .master("local[*]") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

    # Load the merged dataset
    print("Loading merged dataset...")
    try:
        merged_data = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("Data Files/Merged_dataset.csv")

        # Verify that we have the expected columns
        required_columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
        for column in required_columns:
            if column not in merged_data.columns:
                raise Exception(f"Missing required column: {column}")

        # Check for the number of unique genres
        num_genres = merged_data.select("genre").distinct().count()
        print(f"Number of unique genres in the merged dataset: {num_genres}")
        genres = [row.genre for row in merged_data.select("genre").distinct().collect()]
        print(f"Genres in the merged dataset: {', '.join(genres)}")

    except Exception as e:
        print(f"Error loading merged dataset: {e}")
        return None

    # Prepare the data
    print("Preprocessing data...")
    clean_data = merged_data.withColumn("lyrics", lower(col("lyrics")))

    # Split the data into training and test sets
    training_data, test_data = clean_data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training data size: {training_data.count()}, Test data size: {test_data.count()}")

    # Create the ML pipeline
    print("Building ML pipeline...")
    pipeline = create_pipeline()

    # Train the model
    print("Training model on merged dataset... (this may take a few minutes)")
    model = pipeline.fit(training_data)

    # Evaluate the model
    print("Evaluating model performance...")
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test accuracy (Merged model): {accuracy}")

    # Save the model
    print("Saving merged model...")
    model.write().overwrite().save("Generated Directories/trained_model")
    print("Merged model training complete!")

    return model

def predict_genre(lyrics, model_path="Generated Directories/trained_model"):
    """
    Predicts the genre of the given lyrics using the trained model.
    Returns a dictionary of genre probabilities.
    """
    # Initialize Spark session with compatibility configuration for macOS
    try:
        spark = SparkSession.builder \
            .appName("Music Genre Prediction") \
            .master("local[*]") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("\nTrying alternative Spark initialization...")

        # Fallback to minimal configuration
        spark = SparkSession.builder \
            .appName("Music Genre Prediction") \
            .master("local[*]") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

    # Load the model
    try:
        model = PipelineModel.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": "Failed to load model"}

    # Create a DataFrame with the lyrics
    lyrics_df = spark.createDataFrame([(lyrics,)], ["lyrics"])

    # Make prediction
    try:
        prediction = model.transform(lyrics_df)

        # Get the probability distribution
        probabilities = prediction.select("probability").first()[0]

        # Get the label metadata from the model
        label_indexer_model = None
        for stage in model.stages:
            if isinstance(stage, StringIndexerModel):
                label_indexer_model = stage
                break

        if label_indexer_model is None:
            raise Exception("Could not find StringIndexerModel in pipeline stages")

        labels = label_indexer_model.labels

        # Create a dictionary of genre -> probability
        genre_probabilities = {label: float(prob) for label, prob in zip(labels, probabilities)}

        return genre_probabilities
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {"error": "Failed to make prediction"}

def merge_datasets():
    """
    Merges the Mendeley dataset and Student dataset to create the Merged dataset.
    """
    import pandas as pd

    try:
        # Create directories if they don't exist
        os.makedirs("Data Files", exist_ok=True)
        os.makedirs("Generated Directories", exist_ok=True)

        # Check if files exist
        if not os.path.exists("Data Files/Mendeley_dataset.csv"):
            print("Error: Data Files/Mendeley_dataset.csv not found")
            return False

        if not os.path.exists("Data Files/Student_dataset.csv"):
            print("Error: Data Files/Student_dataset.csv not found")
            return False

        print("Loading datasets...")
        mendeley = pd.read_csv("Data Files/Mendeley_dataset.csv")
        student = pd.read_csv("Data Files/Student_dataset.csv")

        # Ensure we only keep the required columns
        columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]

        # Check if required columns exist in both datasets
        for col in columns:
            if col not in mendeley.columns:
                print(f"Error: Column '{col}' missing from Mendeley dataset")
                return False
            if col not in student.columns:
                print(f"Error: Column '{col}' missing from Student dataset")
                return False

        mendeley = mendeley[columns]
        student = student[columns]

        # Merge datasets
        print("Merging datasets...")
        merged = pd.concat([mendeley, student], ignore_index=True)

        # Save merged dataset
        merged.to_csv("Data Files/Merged_dataset.csv", index=False)
        print(f"Merged dataset created with {len(merged)} rows")

        # Report genre counts
        genre_counts = merged['genre'].value_counts()
        print("Genre distribution:")
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} songs")

        return True
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return False

def train_models():
    """Main function to train both models in sequence as required."""
    # First, train on Mendeley dataset (7 genres)
    mendeley_model = train_mendeley_model()
    if mendeley_model is None:
        print("Failed to train Mendeley model. Exiting.")
        return False

    # Then, train on Merged dataset (8 genres)
    merged_model = train_merged_model()
    if merged_model is None:
        print("Failed to train Merged model. Exiting.")
        return False

    print("\n=== MODEL TRAINING COMPLETE ===")
    print("Two models have been created:")
    print("1. mendeley_model - Trained on original 7 genres")
    print("2. trained_model - Trained on all 8 genres (including your custom genre)")

    return True

if __name__ == "__main__":
    # Check if we need to merge datasets
    if not os.path.exists("Data Files/Merged_dataset.csv"):
        print("Merged dataset not found. Creating it now...")
        success = merge_datasets()
        if not success:
            print("Failed to create merged dataset. Exiting.")
            exit(1)

    # Check if both models already exist
    if os.path.exists("Generated Directories/mendeley_model") and os.path.exists("Generated Directories/trained_model"):
        print("Both models already exist. Using existing models.")
    else:
        print("Training models...")
        success = train_models()
        if not success:
            print("Failed to train models. Exiting.")
            exit(1)