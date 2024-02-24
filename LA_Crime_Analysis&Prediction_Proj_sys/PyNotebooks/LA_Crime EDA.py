# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.functions import count, col
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

# COMMAND ----------

la_data= spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ivanfrancis2803@gmail.com/LA_CrimeClean.csv")

# COMMAND ----------

la_data.printSchema()

# COMMAND ----------

top_crm_cd_desc = la_data.groupBy("Crm Cd Desc").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(10)
display(top_crm_cd_desc)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The provided bar chart illustrates the frequency of various crimes in Los Angeles from 2020 till date. Assault is the most commonly committed crime, with counts surpassing 160000 incidents while theft and rape are incidentally less frequent. 

# COMMAND ----------

top_weapon_desc = la_data.groupBy("Weapon Desc").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(10)
display(top_weapon_desc)

# COMMAND ----------

# MAGIC %md
# MAGIC This bar chart represent the top 10 weapon types used for committing the crimes. The weapons include firearms, knives, objects, pepper spray, physical means, pistols, vehicles, verbal threats and few are unknown. In most crimes, criminals tend to attack the victims physically. 

# COMMAND ----------

crime_by_areas = la_data.groupBy("AREA NAME").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(10)
display(crime_by_areas)

# COMMAND ----------

# MAGIC %md
# MAGIC This chart likely represents crime counts by different areas and precincts. The bars represent various locations such as 77th Street, Central, Southeast, and others, with the length of each bar corresponding to the crime count in that area. 

# COMMAND ----------

race_ethnicity = la_data.groupBy("Vict Descent").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(10)
display(race_ethnicity)

# COMMAND ----------

# MAGIC %md
# MAGIC This graph shows the crime count by victim demographic. The categories listed are various ethnicities or racial identifications such as Hispanic/Latin, Black, White, and other classifications, with their corresponding crime counts. As we see, majority of the victims are Latin Americans while the crimes committed on Chinese and American Indians are significantly low.

# COMMAND ----------

gender_crime_count = la_data.groupBy('Vict Sex').agg(count("*").alias("crime count"))
display(gender_crime_count)

# COMMAND ----------

# MAGIC %md
# MAGIC Bar chart represents the crime count based on the sex of the victim, with categories Male, Female, and few are unknown. The chart shows the number of crimes where each group has been the victim. As we can see, both males and females are equally victimised.

# COMMAND ----------

la_data = la_data.withColumn("Vict Age", col("Vict Age").cast("int"))

vict_age_dist = la_data.groupBy("Vict Age").agg(count("*").alias("count")).orderBy(col("count").desc())

display(vict_age_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC It depicts a histogram of the victim age distribution for crimes. The x-axis represents the age of victims, and the y-axis represents the crime count. The distribution appears to show that crime victimization is higher among younger age groups and tends to decrease as age increases. The most common victim age lies between 25 - 35 years of age based on the crimes committed in the dataset.

# COMMAND ----------

vict_ase = la_data.groupBy("Vict Age","Vict Sex").agg(count("*").alias("count")).orderBy(col("count").desc())
display(vict_ase)

# COMMAND ----------

# MAGIC %md 
# MAGIC Line graph shows the victim age distribution by sex for crime incidents. The graph has three lines representing Male, Female, and Unknown, plotted against the victim age on the x-axis and the crime count on the y-axis.
# MAGIC
# MAGIC We can observe that female victims are greater than male victims before the age of 30 . However, it is the opposite as we progress further in the infographic.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

la_data = la_data.withColumn("Year", F.year(F.to_date("Date OCC", "MM-dd-yyyy")))
la_data = la_data.withColumn("Month", F.month(F.to_date("Date OCC", "MM-dd-yyyy")))

yearly_monthly_trend = la_data.groupBy("Year").count().orderBy("Year")
yearly_monthly_trend = yearly_monthly_trend.filter(col("Year") != "2023")
display(yearly_monthly_trend)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC This graph represents the yearly pattern of crimes committed from 2020 to 2023. The graph is likely to be demonstrating the number of crimes each year, with a notable trend or change in crime numbers over these years.

# COMMAND ----------

la_data = la_data.withColumn("Date OCC", F.to_date("Date OCC", "MM-dd-yyyy"))
la_data = la_data.withColumn("DayOfWeek", F.date_format("Date OCC", "EEEE"))
custom_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
order_mapping = spark.createDataFrame([(day, index) for index, day in enumerate(custom_order)], ["DayOfWeek", "Order"])
daily_trend = la_data.groupBy("DayOfWeek").count().join(order_mapping, "DayOfWeek").orderBy("Order").drop("Order")
display(daily_trend)

# COMMAND ----------

plt.figure(figsize=(8, 3))
sns.lineplot(x="DayOfWeek", y="count", data=daily_trend.toPandas())  # Convert to Pandas DataFrame
plt.title("Daily Trend of Crimes 2020-Present")
plt.xlabel("Day of the Week")
plt.ylabel("Crime Count")
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Line graph shows the crime count for each day of the week. It appears to show that majority of crimes are committed during the weekends.

# COMMAND ----------

start_date = "2020-01-01"
end_date = "2023-01-01"
filtered_data = la_data.filter((col("Date OCC") >= start_date) & (col("Date OCC") < end_date))
# Extract the month and create the monthly trend DataFrame
monthly_trend = filtered_data.withColumn("Month", month("Date OCC")).groupBy("Month").count().orderBy("Month")
# Display the DataFrame

display(monthly_trend)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Line graph illustrates the monthly trend of crimes. The x-axis lists the months from January to December, and the y-axis represents the crime count. The line graph indicates changes in the number of crimes reported each month. There appears to be a peak in crime count around mid-year, with a decline towards the end of the year.

# COMMAND ----------

from pyspark.sql.window import Window

# COMMAND ----------

latin_data = la_data.filter(F.col("Vict Descent") == "Hispanic/Latin")
latin_data = latin_data.groupBy("AREA NAME").count()
display(latin_data)

# COMMAND ----------

# MAGIC %md
# MAGIC The horizontal bar chart shows the crime count against Hispanic or Latin victims across various areas or precincts in LA. Each bar represents a different area, with the length of the bar correlating to the number of crimes committed against victims of Hispanic or Latin descent in that specific area.

# COMMAND ----------

Bl_data = la_data.filter(F.col("Vict Descent") == "Black")
Bl_data = Bl_data.groupBy("AREA NAME").count()
display(Bl_data)

# COMMAND ----------

# MAGIC %md
# MAGIC horizontal bar chart displays the number of crimes committed against Black victims in different areas or precincts. Each bar corresponds to a unique area, labeled on the y-axis, and extends horizontally to indicate the count of crimes, which is represented on the x-axis. The lengths of the bars are proportional to the crime count.

# COMMAND ----------

wht_data = la_data.filter(F.col("Vict Descent") == "White")
wht_data = wht_data.groupBy("AREA NAME").count()
display(wht_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Horizontal bar chart presents the number of crimes committed against White victims in different areas or precincts. Each bar represents a unique area, labeled on the y-axis, with the length of the bar corresponding to the crime count against White victims, as shown on the x-axis. The chart shows that areas such as Hollywood, Pacific and Central LA experience more crimes.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Research Question 1: How do crime rates against different ethnic groups vary across different areas, and what might this indicate about the socio-economic and cultural dynamics in those areas?
# MAGIC
# MAGIC The bar charts indicate that crime rates against Hispanic/Latin, Black, and White victims vary significantly across different areas. For instance, some areas might show a higher crime rate against Hispanic/Latin individuals, which could be indicative of larger Hispanic/Latin populations in those areas or may reflect targeted criminal activity. Similarly, areas with higher crime rates against Black victims might correlate with socio-economic factors such as poverty, unemployment, or educational disparities that are known to affect crime rates. For White victims, the distribution of crime might relate to the demographic makeup or could point towards specific types of crimes that are more prevalent in certain areas. This variation in crime rates could be a reflection of underlying issues such as segregation, economic disparities, and varying levels of police presence or community resources. Analyzing these trends could inform more culturally and socially sensitive policing and community support services, and might also guide initiatives to address underlying causes of crime such as inequality or lack of social mobility. 
# MAGIC
# MAGIC Research Question 2: What does the monthly and daily trend of crimes suggest about the effectiveness of law enforcement strategies and the need for community intervention programs?
# MAGIC
# MAGIC  The monthly crime trends graph showed fluctuations with peaks and troughs which could correspond with seasonal activities, police enforcement efficacy, community events, or even climatic conditions. A peak in mid-year might coincide with summer breaks when there is an increase in outdoor activities, potentially leading to more opportunities for crime. A decline towards the end of the year might be due to the colder weather or increased end-of-year policing efforts. The daily crime trend graph highlighted variations in crime rates across different days of the week. A potential increase in crimes on weekends could suggest that there are more opportunities for criminals when people are off work and more public events are taking place. Alternatively, it might reflect differing police patrol schedules or community vigilance levels on these days. These patterns suggest that law enforcement strategies could be fine-tuned to be more proactive during periods of expected higher crime rates. Community intervention programs could also be scheduled to coincide with these periods to engage at-risk populations, provide alternative activities for youth, or increase community vigilance and awareness. Such data-driven strategies could enhance the overall effectiveness of crime prevention and community safety initiatives.
# MAGIC