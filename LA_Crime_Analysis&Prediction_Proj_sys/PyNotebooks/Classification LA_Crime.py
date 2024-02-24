# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ivanfrancis2803@gmail.com/LACrimePre.csv")

# COMMAND ----------

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in ['Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc', 'LOCATION']]

encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in ['Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc', 'LOCATION']]

from pyspark.sql.types import IntegerType, DoubleType
df = df.withColumn("Vict Age", df["Vict Age"].cast(IntegerType()))
df = df.withColumn("LAT", df["LAT"].cast(DoubleType()))
df = df.withColumn("LON", df["LON"].cast(DoubleType()))
df = df.withColumn("Crm Cd", df["Crm Cd"].cast(IntegerType()))
df = df.withColumn("Premis Cd", df["Premis Cd"].cast(IntegerType()))
df = df.withColumn("Rpt Dist No", df["Rpt Dist No"].cast(IntegerType()))
df = df.withColumn("Weapon Used Cd", df["Weapon Used Cd"].cast(IntegerType()))

# COMMAND ----------

print(df.schema)

# COMMAND ----------

selected_features = ['Vict Age', 'Vict Sex_encoded', 'Vict Descent_encoded', 'Crm Cd', 'Premis Cd', 'Rpt Dist No', 'Premis Desc_encoded', 'Weapon Used Cd', 'LOCATION_encoded']

# COMMAND ----------

label_indexer = StringIndexer(inputCol='Status Desc', outputCol="label", handleInvalid="skip")

# COMMAND ----------

feature_assembler = VectorAssembler(inputCols=selected_features, outputCol="features")

# COMMAND ----------

rf_classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

# COMMAND ----------

pipeline = Pipeline(stages=indexers + encoders + [label_indexer, feature_assembler, rf_classifier])

# COMMAND ----------

(training_data, testing_data) = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

model = pipeline.fit(training_data)

# COMMAND ----------

predictions = model.transform(testing_data)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# COMMAND ----------

feature_importances = model.stages[-1].featureImportances
print("Feature Importances:")
for i, feature in enumerate(selected_features):
    print(f"{feature}: {feature_importances[i]}")

# COMMAND ----------

selected_columns = ['Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc', 'LOCATION', 'Status Desc', 'prediction']

predictions.select(selected_columns).show(truncate=False)

# COMMAND ----------

