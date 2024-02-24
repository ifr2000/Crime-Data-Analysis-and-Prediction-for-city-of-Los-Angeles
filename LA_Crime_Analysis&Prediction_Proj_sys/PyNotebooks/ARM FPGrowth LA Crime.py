# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

# COMMAND ----------

LA_data = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ivanfrancis2803@gmail.com/LA_CrimeClean.csv")

# COMMAND ----------

LA_data.count()

# COMMAND ----------

LA_data.show()

# COMMAND ----------

la_apr = ['AREA NAME', 'Crm Cd Desc', 'Premis Desc', 'Weapon Desc', 'Vict Descent']
La_data_apr = LA_data.select(la_apr)
La_data_apr.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Deleting Unknowns and Others because while creating item set FPGrowth algorithm needs all values in the list unique
# MAGIC

# COMMAND ----------

La_data_apr = La_data_apr.filter(
    ~(
        (col("AREA NAME") == "Unknown") |
        (col("Crm Cd Desc") == "Unknown") |
        (col("Premis Desc") == "Unknown") |
        (col("Weapon Desc") == "Unknown") |
        (col("Vict Descent") == "Unknown") 
    )
)

La_data_apr = La_data_apr.filter(
    ~(
        (col("AREA NAME") == "Other") |
        (col("Crm Cd Desc") == "Other") |
        (col("Premis Desc") == "Other") |
        (col("Weapon Desc") == "Other") |
        (col("Vict Descent") == "Other")
    )
)

La_data_apr.count()

# COMMAND ----------

from pyspark.sql.functions import array, col, expr


item_columns = La_data_apr.columns


transformed_data = La_data_apr.withColumn(
    "Crimes",
    array(*[col(column) for column in item_columns]) 
)


La_df = transformed_data.withColumn(
    "Crimes",
    expr("filter(Crimes, x -> x is not null)")
)


selected_columns = ["AREA NAME", "Crm Cd Desc", "Premis Desc", "Weapon Desc", "Vict Descent"]
La_df = La_df.select(selected_columns + ["Crimes"])

La_df.show(truncate=False)

# COMMAND ----------

La_df=La_df.drop("AREA NAME","Crm Cd Desc","Premis Desc","Weapon Desc","Vict Descent")
La_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC splitting the data into train and test ( 80 / 20)

# COMMAND ----------

from pyspark.ml.fpm import FPGrowth

train_data, test_data = La_df.randomSplit([0.8, 0.2], seed=123)

# COMMAND ----------

fp_growth = FPGrowth(itemsCol="Crimes", minSupport=0.2, minConfidence=0.5)
La_model = fp_growth.fit(train_data)

# Display frequent itemsets.
La_model.freqItemsets.show()

# Display generated association rules.
La_model.associationRules.show(20,truncate=False)
La_model.transform(test_data).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Research Question:
# MAGIC How does the association rule mining output reveal patterns in the relationship between crime characteristics and demographics, and what insights can be drawn regarding crime occurrence in specific demographic groups within different areas?
# MAGIC
# MAGIC Answer:
# MAGIC The association rule mining output provides valuable insights into the relationships between crime characteristics and demographics in the given dataset.
# MAGIC
# MAGIC Frequent Itemsets:
# MAGIC The most frequent crime itemset is [Assault] with a frequency of 113,421.
# MAGIC The itemset [Hispanic/Latin] appears frequently (85,298) in the dataset, indicating a significant presence of this demographic factor.
# MAGIC Itemsets involving [Physical] and [Residential Area] also have high frequencies, suggesting common occurrences of physical crimes that include bodily force or battery inflicting physical injury in residential areas. 
# MAGIC
# MAGIC Association Rules:
# MAGIC The association rules highlight the confidence, lift, and support values for different combinations of antecedents and consequents.
# MAGIC For example, the rule [Hispanic/Latin] -> [Physical] has a confidence of 63.05%, suggesting that 63.05% of the occurrences of Hispanic/Latin demographic factor are associated with physical crimes.The lift values indicate how much more likely the consequent is to occur given the antecedent.The high confidence value signifies a reasonably strong association between crimes with a physical nature and the demographic factor of Hispanic/Latin descent.
# MAGIC
# MAGIC Also, the rule [Hispanic/Latin] -> [Physical] has a lift value of 1.01, suggesting a slightly increased likelihood of physical crimes occurring when the demographic factor is Hispanic/Latin compared to random chance.
# MAGIC
# MAGIC Predictions:
# MAGIC The prediction column shows the predicted itemsets based on the association rules for a given set of crimes.
# MAGIC For instance, the prediction [Physical, Hispanic/Latin] suggests a potential association between physical crimes that include bodily force or battery inflicting physical injury and the Hispanic/Latin demographic factor.