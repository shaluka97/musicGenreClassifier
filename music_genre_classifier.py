import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lower, col

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'
os.environ['JAVA_OPTS'] = '-Djava.security.manager=allow'

try:
    import findspark
    findspark.init()
except ImportError:
    print("Note: findspark not found. This is fine if your PySpark setup is already configured.")

def create_pipeline():
    tokenizer = RegexTokenizer(inputCol="lyrics", outputCol="words", pattern="\\W+")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    indexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="keep")
    classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    return Pipeline(stages=[tokenizer, remover, hashing_tf, idf, indexer, classifier])

def train_merged_model():
    print("\n=== TRAINING MODEL ON MERGED DATASET ===\n")
    print("Initializing Spark session...")
    try:
        spark = SparkSession.builder \
            .appName("Music Genre Classification") \
            .master("local[*]") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("\nTrying alternative Spark initialization...")

        spark = SparkSession.builder \
            .appName("Music Genre Classification") \
            .master("local[*]") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

    print("Loading merged dataset...")
    try:
        merged_data = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("Merged_dataset.csv")

        required_columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
        for column in required_columns:
            if column not in merged_data.columns:
                raise Exception(f"Missing required column: {column}")

        num_genres = merged_data.select("genre").distinct().count()
        print(f"Number of unique genres in the merged dataset: {num_genres}")
        genres = [row.genre for row in merged_data.select("genre").distinct().collect()]
        print(f"Genres in the merged dataset: {', '.join(genres)}")

    except Exception as e:
        print(f"Error loading merged dataset: {e}")
        return None

    print("Preprocessing data...")
    clean_data = merged_data.withColumn("lyrics", lower(col("lyrics")))

    training_data, test_data = clean_data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training data size: {training_data.count()}, Test data size: {test_data.count()}")

    print("Building ML pipeline...")
    pipeline = create_pipeline()

    print("Training model on merged dataset... (this may take a few minutes)")
    model = pipeline.fit(training_data)

    print("Evaluating model performance...")
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test accuracy: {accuracy}")

    print("Saving model...")
    os.makedirs("Trained_Final_Model", exist_ok=True)
    model.write().overwrite().save("Trained_Final_Model")
    print("Model training complete!")

    return model

def predict_genre(lyrics, model_path="Trained_Final_Model"):
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

        spark = SparkSession.builder \
            .appName("Music Genre Prediction") \
            .master("local[*]") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

    try:
        model = PipelineModel.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": "Failed to load model"}

    lyrics_df = spark.createDataFrame([(lyrics,)], ["lyrics"])

    try:
        prediction = model.transform(lyrics_df)

        probabilities = prediction.select("probability").first()[0]

        label_indexer_model = None
        for stage in model.stages:
            if isinstance(stage, StringIndexerModel):
                label_indexer_model = stage
                break

        if label_indexer_model is None:
            raise Exception("Could not find StringIndexerModel in pipeline stages")

        labels = label_indexer_model.labels

        genre_probabilities = {label: float(prob) for label, prob in zip(labels, probabilities)}

        return genre_probabilities
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {"error": "Failed to make prediction"}

def merge_datasets():
    import pandas as pd

    try:
        if not os.path.exists("Mendeley_dataset.csv"):
            print("Error: Mendeley_dataset.csv not found")
            return False

        if not os.path.exists("Student_dataset.csv"):
            print("Error: Student_dataset.csv not found")
            return False

        print("Loading datasets...")
        mendeley = pd.read_csv("Mendeley_dataset.csv")
        student = pd.read_csv("Student_dataset.csv")

        columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]

        for col in columns:
            if col not in mendeley.columns:
                print(f"Error: Column '{col}' missing from Mendeley dataset")
                return False
            if col not in student.columns:
                print(f"Error: Column '{col}' missing from Student dataset")
                return False

        mendeley = mendeley[columns]
        student = student[columns]

        print("Merging datasets...")
        merged = pd.concat([mendeley, student], ignore_index=True)

        merged.to_csv("Merged_dataset.csv", index=False)
        print(f"Merged dataset created with {len(merged)} rows")

        genre_counts = merged['genre'].value_counts()
        print("Genre distribution:")
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} songs")

        return True
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists("Merged_dataset.csv"):
        print("Merged dataset not found. Creating it now...")
        success = merge_datasets()
        if not success:
            print("Failed to create merged dataset. Exiting.")
            exit(1)

    if os.path.exists("Trained_Final_Model"):
        print("Model already exists. Using existing model.")
    else:
        print("Training model...")
        model = train_merged_model()
        if model is None:
            print("Failed to train model. Exiting.")
            exit(1)