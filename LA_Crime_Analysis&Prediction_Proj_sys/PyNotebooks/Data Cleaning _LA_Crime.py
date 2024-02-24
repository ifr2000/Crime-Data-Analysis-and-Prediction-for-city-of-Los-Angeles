# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.functions import from_unixtime, unix_timestamp, date_format
from pyspark.sql.functions import when, col

# COMMAND ----------

# MAGIC %md 
# MAGIC We are working with the LA_Crime data from their open portal "data.lacity.org" with approximately 750,000 records and 18 original data fields
# MAGIC

# COMMAND ----------

LA_Raw = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ivanfrancis2803@gmail.com/LA_Crime.csv")

# COMMAND ----------

LA_Raw.count()

# COMMAND ----------

LA_Raw.show()

# COMMAND ----------

missing_values_count = LA_Raw.select([sum(col(c).isNull().cast("int")).alias(c) for c in LA_Raw.columns])

# COMMAND ----------

missing_values_count.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC We dropped the column values that were not going to contribute towards the analysis as Codes for weapons or crime description were redundant because categorical values for everyone one of them was provided.

# COMMAND ----------

drop_col = ['DR_NO','AREA','Rpt Dist No','Part 1-2','Crm Cd','Mocodes','Premis Cd','Weapon Used Cd','Status','Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4','Cross Street']

LA_Raw = LA_Raw.drop(*drop_col)

# COMMAND ----------

# MAGIC %md
# MAGIC Converting the Victim Age column to numerical type

# COMMAND ----------

LA_Raw=LA_Raw.withColumn('Vict Age',col('Vict Age').cast('int'))

# COMMAND ----------

df_transformed = LA_Raw.withColumn("Date Rptd", substring("Date Rptd", 1, 10))
df_transformed = df_transformed.withColumn("Date OCC", substring("Date OCC", 1, 10))

# COMMAND ----------

df_transformed = df_transformed.withColumn("Date Rptd", regexp_replace("Date Rptd", "/", "-")) \
                  .withColumn("Date OCC", regexp_replace("Date OCC", "/", "-"))

# COMMAND ----------

# MAGIC %md
# MAGIC Converting time to a consistent format 

# COMMAND ----------

df_transformed = df_transformed.withColumn("TIME OCC", lpad(col("TIME OCC"), 4, "0"))

# COMMAND ----------

df_transformed = df_transformed.withColumn(
    "TIME OCC", 
    concat(substring(col("TIME OCC"), 1, 2), lit(":"), substring(col("TIME OCC"), 3, 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Mapping the victim Descent and Sex and converting them to description that is understandable.

# COMMAND ----------

victim_sex_mapping = {'F': 'Female', 'M': 'Male', 'X': 'Unknown'}
victim_desc_mapping = {
    'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian', 'F': 'Filipino',
    'G': 'Guamanian', 'H': 'Hispanic/Latin', 'I': 'American Indian/Alaskan Native',
    'J': 'Japanese', 'K': 'Korean', 'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander',
    'S': 'Samoan', 'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown',
    'Z': 'Asian Indian'
}


# COMMAND ----------

df_transformed = df_transformed.withColumn("Vict Sex", col("Vict Sex").cast("string"))  
for key, value in victim_sex_mapping.items():
    df_transformed = df_transformed.replace(key, value, "Vict Sex")

# COMMAND ----------


df_transformed = df_transformed.withColumn("Vict Descent", col("Vict Descent").cast("string"))  
for key, value in victim_desc_mapping.items():
    df_transformed = df_transformed.replace(key, value, "Vict Descent")

# COMMAND ----------

df_transformed.select("Crm Cd Desc").distinct().show(1000,truncate=False)

# COMMAND ----------

df_transformed.select("Premis Desc").distinct().show(1000,truncate=False)

# COMMAND ----------

df_transformed.select("Weapon Desc").distinct().show(1000,truncate=False)

# COMMAND ----------

df_transformed.select("Status Desc").distinct().show(1000,truncate=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC While observing the data column for Crime, Weapon and Premise Description, it was observed that the grouping similar crimes , premises or weapons used under broader categories simplifies the analysis of crime data. It allows for a higher-level overview of the types of crimes being committed without getting lost in the details of every specific offense.
# MAGIC
# MAGIC Which is what is carried out further:
# MAGIC for eg in weapon description,
# MAGIC UZI SEMIAUTOMATIC ASSAULT RIFLE
# MAGIC HECKLER & KOCH 91 SEMIAUTOMATIC ASSAULT RIFLE 
# MAGIC
# MAGIC are both mapped as "Semi Automatic Rifle"

# COMMAND ----------

crime_mapping = {
    "GRAND THEFT / INSURANCE FRAUD": "Fraud",
    "VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)": "Vandalism",
    "RECKLESS DRIVING": "Traffic Violation",
    "DRIVING WITHOUT OWNER CONSENT (DWOC)": "Vehicle Theft",
    "DISCHARGE FIREARMS/SHOTS FIRED": "Discharge of Firearms",
    "THROWING OBJECT AT MOVING VEHICLE": "Assault",
    "PROWLER": "Trespassing",
    "BURGLARY FROM VEHICLE, ATTEMPTED": "Burglary",
    "SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH": "Sexual Assault",
    "FIREARMS RESTRAINING ORDER (FIREARMS RO)": "Firearms Restraining Order",
    "INDECENT EXPOSURE": "Indecent Exposure",
    "CRIMINAL HOMICIDE": "Homicide",
    "THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)": "Vehicle Theft",
    "STALKING": "Stalking",
    "FAILURE TO YIELD": "Traffic Violation",
    "EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)": "Fraud",
    "CRIMINAL THREATS - NO WEAPON DISPLAYED": "Assault",
    "ORAL COPULATION": "Sexual Assault",
    "BATTERY - SIMPLE ASSAULT": "Assault",
    "THREATENING PHONE CALLS/LETTERS": "Harassment",
    "UNAUTHORIZED COMPUTER ACCESS": "Cybercrime",
    "LEWD/LASCIVIOUS ACTS WITH CHILD": "Child Endangerment",
    "BUNCO, ATTEMPT": "Fraud",
    "TILL TAP - GRAND THEFT ($950.01 & OVER)": "Theft",
    "THEFT, COIN MACHINE - PETTY ($950 & UNDER)": "Theft",
    "DOCUMENT FORGERY / STOLEN FELONY": "Forgery",
    "ARSON": "Arson",
    "CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT": "Child Abuse",
    "ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER": "Assault",
    "INTIMATE PARTNER - AGGRAVATED ASSAULT": "Assault",
    "CHILD NEGLECT (SEE 300 W.I.C.)": "Child Neglect",
    "PEEPING TOM": "Privacy Violation",
    "EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)": "Fraud",
    "CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT": "Assault",
    "LEWD CONDUCT": "Indecent Exposure",
    "CREDIT CARDS, FRAUD USE ($950 & UNDER": "Credit Card Fraud",
    "TILL TAP - PETTY ($950 & UNDER)": "Theft",
    "DISRUPT SCHOOL": "Disruption of School",
    "CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)": "Child Abuse",
    "BRIBERY": "Bribery",
    "BIKE - STOLEN": "Theft",
    "BATTERY WITH SEXUAL CONTACT": "Sexual Assault",
    "DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01": "Theft",
    "SEXUAL PENETRATION W/FOREIGN OBJECT": "Sexual Assault",
    "COUNTERFEIT": "Forgery",
    "DISHONEST EMPLOYEE - GRAND THEFT": "Fraud",
    "PICKPOCKET": "Theft",
    "PIMPING": "Prostitution",
    "RAPE, FORCIBLE": "Rape",
    "CHILD STEALING": "Child Abduction",
    "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE": "Human Trafficking",
    "BUNCO, GRAND THEFT": "Fraud",
    "VEHICLE - ATTEMPT STOLEN": "Attempted Vehicle Theft",
    "VIOLATION OF COURT ORDER": "Violation of Court Order",
    "FALSE POLICE REPORT": "False Report",
    "VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)": "Vehicle Theft",
    "LYNCHING": "Assault",
    "SHOPLIFTING - PETTY THEFT ($950 & UNDER)": "Shoplifting",
    "FALSE IMPRISONMENT": "False Imprisonment",
    "CHILD PORNOGRAPHY": "Child Exploitation",
    "CREDIT CARDS, FRAUD USE ($950.01 & OVER)": "Credit Card Fraud",
    "BUNCO, PETTY THEFT": "Fraud",
    "PURSE SNATCHING - ATTEMPT": "Attempted Theft",
    "RAPE, ATTEMPTED": "Attempted Rape",
    "DOCUMENT WORTHLESS ($200.01 & OVER)": "Forgery",
    "THEFT, COIN MACHINE - GRAND ($950.01 & OVER)": "Theft",
    "BATTERY POLICE (SIMPLE)": "Assault",
    "EXTORTION": "Extortion",
    "INTIMATE PARTNER - SIMPLE ASSAULT": "Assault",
    "THEFT PLAIN - PETTY ($950 & UNDER)": "Theft",
    "VIOLATION OF RESTRAINING ORDER": "Violation of Restraining Order",
    "ATTEMPTED ROBBERY": "Attempted Robbery",
    "SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ": "Sexual Offense",
    "CHILD ANNOYING (17YRS & UNDER)": "Child Annoyance",
    "CONTEMPT OF COURT": "Contempt of Court",
    "MANSLAUGHTER, NEGLIGENT": "Manslaughter",
    "DISTURBING THE PEACE": "Disturbing the Peace",
    "WEAPONS POSSESSION/BOMBING": "Weapons Offense",
    "ROBBERY": "Robbery",
    "DISHONEST EMPLOYEE - PETTY THEFT": "Fraud",
    "HUMAN TRAFFICKING - COMMERCIAL SEX ACTS": "Human Trafficking",
    "ILLEGAL DUMPING": "Environmental Offense",
    "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT": "Assault",
    "SHOPLIFTING-GRAND THEFT ($950.01 & OVER)": "Shoplifting",
    "RESISTING ARREST": "Resisting Arrest",
    "VEHICLE - STOLEN": "Vehicle Theft",
    "BRANDISH WEAPON": "Brandishing",
    "TRESPASSING": "Trespassing",
    "DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER": "Theft",
    "BURGLARY FROM VEHICLE": "Burglary",
    "THEFT FROM PERSON - ATTEMPT": "Attempted Theft",
    "PETTY THEFT - AUTO REPAIR": "Theft",
    "THEFT PLAIN - ATTEMPT": "Attempted Theft",
    "DOCUMENT WORTHLESS ($200 & UNDER)": "Forgery",
    "BOAT - STOLEN": "Theft",
    "OTHER ASSAULT": "Assault",
    "KIDNAPPING - GRAND ATTEMPT": "Attempted Kidnapping",
    "BOMB SCARE": "Bomb Threat",
    "SEX OFFENDER REGISTRANT OUT OF COMPLIANCE": "Non-Compliance",
    "THEFT, COIN MACHINE - ATTEMPT": "Attempted Theft",
    "THEFT FROM MOTOR VEHICLE - ATTEMPT": "Attempted Theft",
    "SHOPLIFTING - ATTEMPT": "Attempted Shoplifting",
    "BIKE - ATTEMPTED STOLEN": "Attempted Theft",
    "OTHER MISCELLANEOUS CRIME": "Miscellaneous",
    "THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD": "Theft",
    "THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)": "Vehicle Theft",
    "SHOTS FIRED AT INHABITED DWELLING": "Discharge of Firearms",
    "LETTERS, LEWD  -  TELEPHONE CALLS, LEWD": "Harassment",
    "SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT": "Discharge of Firearms",
    "BURGLARY, ATTEMPTED": "Attempted Burglary",
    "LYNCHING - ATTEMPTED": "Attempted Assault",
    "PANDERING": "Prostitution",
    "VIOLATION OF TEMPORARY RESTRAINING ORDER": "Violation of Restraining Order",
    "KIDNAPPING": "Kidnapping",
    "BURGLARY": "Burglary",
    "VANDALISM - MISDEAMEANOR ($399 OR UNDER)": "Vandalism",
    "THEFT, PERSON": "Theft",
    "PURSE SNATCHING": "Theft",
    "THEFT OF IDENTITY": "Identity Theft",
    "CONSPIRACY": "Conspiracy",
    "CRUELTY TO ANIMALS": "Animal Cruelty",
    "BATTERY ON A FIREFIGHTER": "Assault",
    "CONTRIBUTING": "Contributing to Delinquency",
    "DRUNK ROLL": "Theft",
    "REPLICA FIREARMS(SALE,DISPLAY,MANUFACTURE OR DISTRIBUTE)": "Weapons Offense",
    "CHILD ABANDONMENT": "Child Neglect",
    "GRAND THEFT / AUTO REPAIR": "Fraud",
    "PICKPOCKET, ATTEMPT": "Attempted Theft",
    "DRUGS, TO A MINOR": "Drug Offense",
    "BIGAMY": "Bigamy",
    "FIREARMS EMERGENCY PROTECTIVE ORDER (FIREARMS EPO)": "Firearms Restraining Order",
    "FAILURE TO DISPERSE": "Disorderly Conduct",
    "BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM": "Sexual Offense",
    "TELEPHONE PROPERTY - DAMAGE": "Property Damage",
    "INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)": "Sexual Offense",
    "BLOCKING DOOR INDUCTION CENTER": "Obstruction of Justice",
    "DISHONEST EMPLOYEE ATTEMPTED THEFT": "Fraud",
    "INCITING A RIOT": "Inciting a Riot"
}

# COMMAND ----------

len(crime_mapping)

# COMMAND ----------

df_transformed = df_transformed.withColumn("Crm Cd Desc", col("Crm Cd Desc").cast("string"))  
for key, value in crime_mapping.items():
    df_transformed = df_transformed.replace(key, value, "Crm Cd Desc")

# COMMAND ----------

# MAGIC %md
# MAGIC Mapping the crimes to a much more generalized description we funneled down 138 different crime descriptions to 69 
# MAGIC

# COMMAND ----------

premise_mapping = {
    "BUS STOP": "Public Area",
    "MASSAGE PARLOR": "Commercial Area",
    "RIVER BED*": "Outdoor Area",
    "MOTEL": "Commercial Area",
    "MTA - SILVER LINE - HARBOR GATEWAY TRANSIT CTR": "Public Area",
    "OPTICAL OFFICE INSIDE STORE OR SUPERMARKET*": "Commercial Area",
    "SIDEWALK": "Public Area",
    "SYNAGOGUE/TEMPLE": "Religious Institution",
    "STAIRWELL*": "Indoor Area",
    "APARTMENT/CONDO COMMON LAUNDRY ROOM": "Residential Area",
    "CULTURAL SIGNIFICANCE/MONUMENT": "Landmark",
    "CAR WASH": "Commercial Area",
    "MTA - PURPLE LINE - CIVIC CENTER/GRAND PARK": "Public Area",
    "THE GROVE": "Commercial Area",
    "GOLF COURSE*": "Recreational Area",
    "OTHER BUSINESS": "Commercial Area",
    "TOBACCO SHOP": "Commercial Area",
    "HOSPITAL": "Healthcare Facility",
    "SLIPS/DOCK/MARINA/BOAT": "Waterfront Area",
    "FOSTER HOME BOYS OR GIRLS*": "Residential Area",
    "ABORTION CLINIC/ABORTION FACILITY*": "Healthcare Facility",
    "PAY PHONE": "Public Area",
    "SINGLE RESIDENCE OCCUPANCY (SRO'S) LOCATIONS": "Residential Area",
    "METROLINK TRAIN": "Public Transportation",
    "TRANSIENT ENCAMPMENT": "Outdoor Area",
    "MOBILE HOME/TRAILERS/CONSTRUCTION TRAILERS/RV'S/MOTORHOME": "Residential Area",
    "VACANT LOT": "Outdoor Area",
    "MTA - RED LINE - HOLLYWOOD/WESTERN": "Public Transportation",
    "MTA - SILVER LINE - DOWNTOWN STREET STOPS": "Public Transportation",
    "GREYHOUND OR INTERSTATE BUS": "Public Transportation",
    "DODGER STADIUM": "Sports Venue",
    "DRIVEWAY": "Residential Area",
    "STAPLES CENTER *": "Sports Venue",
    "AUTOMATED TELLER MACHINE (ATM)": "Commercial Area",
    "MUNICIPAL BUS LINE INCLUDES LADOT/DASH": "Public Transportation",
    "SPECIALTY SCHOOL/OTHER": "Educational Institution",
    "WAREHOUSE": "Commercial Area",
    "EQUIPMENT RENTAL": "Commercial Area",
    "MTA - SILVER LINE - ROSECRANS": "Public Transportation",
    "MTA - GREEN LINE - AVIATION/LAX": "Public Transportation",
    "YARD (RESIDENTIAL/BUSINESS)": "Residential Area",
    "PEDESTRIAN OVERCROSSING": "Public Area",
    "TRAIN TRACKS": "Public Transportation",
    "ENERGY PLANT/FACILITY": "Industrial Area",
    "MTA - SILVER LINE - 37TH ST/USC": "Public Transportation",
    "COLISEUM": "Sports Venue",
    "BAR/COCKTAIL/NIGHTCLUB": "Entertainment Venue",
    "CONVENTION CENTER": "Event Venue",
    "THE BEVERLY CENTER": "Commercial Area",
    "AMTRAK TRAIN": "Public Transportation",
    "MISSIONS/SHELTERS": "Shelter",
    "null": "Unknown",
    "ARCADE,GAME ROOM/VIDEO GAMES (EXAMPLE CHUCKIE CHEESE)*": "Entertainment Venue",
    "MTA - GOLD LINE - LITTLE TOKYO/ARTS DISTRICT": "Public Transportation",
    "NURSING/CONVALESCENT/RETIREMENT HOME": "Healthcare Facility",
    "PET STORE": "Commercial Area",
    "POST OFFICE": "Government Facility",
    "BANK": "Financial Institution",
    "ESCALATOR*": "Public Area",
    "MINI-MART": "Commercial Area",
    "TOW YARD*": "Commercial Area",
    "PARK/PLAYGROUND": "Recreational Area",
    "ELEMENTARY SCHOOL": "Educational Institution",
    "MTA - PURPLE LINE - 7TH AND METRO CENTER": "Public Transportation",
    "DELIVERY SERVICE (FED EX, UPS, COURIERS,COURIER SERVICE)*": "Commercial Area",
    "MTA - EXPO LINE - EXPO/CRENSHAW": "Public Transportation",
    "MTA - ORANGE LINE - TAMPA": "Public Transportation",
    "SEX ORIENTED/BOOK STORE/STRIP CLUB/GENTLEMAN'S CLUB": "Adult Entertainment Venue",
    "BASKETBALL COURTS": "Recreational Area",
    "MTA - EXPO LINE - EXPO/BUNDY": "Public Transportation",
    "STUDIO (FILM/PHOTOGRAPHIC/MUSIC)": "Commercial Area",
    "HOSPICE": "Healthcare Facility",
    "THEATRE/MOVIE": "Entertainment Venue",
    "MUSEUM": "Cultural Institution",
    "TAXI": "Transportation",
    "MTA - ORANGE LINE - SEPULVEDA": "Public Transportation",
    "RESTAURANT/FAST FOOD": "Dining Establishment",
    "NURSERY/FLOWER SHOP": "Commercial Area",
    "DAY CARE/ADULTS*": "Day Care Facility",
    "LAUNDROMAT": "Commercial Area",
    "SEWAGE FACILITY/PIPE": "Utility Facility",
    "ELEVATOR": "Indoor Area",
    "TELECOMMUNICATION FACILITY/LOCATION": "Utility Facility",
    "CONDOMINIUM/TOWNHOUSE": "Residential Area",
    "CHECK CASHING*": "Financial Institution",
    "AIRCRAFT": "Transportation",
    "MTA - EXPO LINE - 7TH AND METRO CENTER": "Public Transportation",
    "CYBERSPACE": "Online",
    "TRANSITIONAL HOUSING/HALFWAY HOUSE": "Shelter",
    "MTA - ORANGE LINE - WOODLEY": "Public Transportation",
    "MTA - GOLD LINE - HIGHLAND PARK": "Public Transportation",
    "BAR/SPORTS BAR (OPEN DAY & NIGHT)": "Entertainment Venue",
    "NAIL SALON": "Commercial Area",
    "MTA - RED LINE - VERMONT/SANTA MONICA": "Public Transportation",
    "AMUSEMENT PARK*": "Entertainment Venue",
    "HOTEL": "Hospitality Venue",
    "OTHER RR TRAIN (UNION PAC, SANTE FE ETC": "Public Transportation",
    "OIL REFINERY": "Industrial Area",
    "STREET": "Public Area",
    "DRIVE THRU*": "Dining Establishment",
    "MULTI-UNIT DWELLING (APARTMENT, DUPLEX, ETC)": "Residential Area",
    "CEMETARY*": "Cemetery",
    "ABATEMENT LOCATION": "Unknown",
    "MTA - RED LINE - VERMONT/SUNSET": "Public Transportation",
    "TV/RADIO/APPLIANCE": "Commercial Area",
    "MTA - RED LINE - PERSHING SQUARE": "Public Transportation",
    "MTA - EXPO LINE - EXPO/LA BREA": "Public Transportation",
    "FURNITURE STORE": "Commercial Area",
    "MTA - EXPO LINE - EXPO PARK/USC": "Public Transportation",
    "SPORTS ARENA": "Sports Venue",
    "MTA - GOLD LINE - MARIACHI PLAZA": "Public Transportation",
    "JEWELRY STORE": "Commercial Area",
    "MTA - BLUE LINE - PICO": "Public Transportation",
    "MTA - ORANGE LINE - RESEDA": "Public Transportation",
    "HIGH SCHOOL": "Educational Institution",
    "OFFICE BUILDING/OFFICE": "Commercial Area",
    "BOWLING ALLEY*": "Recreational Area",
    "LIBRARY": "Educational Institution",
    "MTA - EXPO LINE - FARMDALE": "Public Transportation",
    "7TH AND METRO CENTER (NOT LINE SPECIFIC)": "Public Transportation",
    "TRANSPORTATION FACILITY (AIRPORT)": "Transportation",
    "MEMBERSHIP STORE (COSTCO,SAMS CLUB)*": "Commercial Area",
    "COLLEGE/JUNIOR COLLEGE/UNIVERSITY": "Educational Institution",
    "PUBLIC RESTROOM/OUTSIDE*": "Public Area",
    "MTA - SILVER LINE - UNION STATION": "Public Transportation",
    "MTA - GOLD LINE - PICO/ALISO": "Public Transportation",
    "MTA - RED LINE - VERMONT/BEVERLY": "Public Transportation",
    "CREDIT UNION": "Financial Institution",
    "MTA - RED LINE - HOLLYWOOD/HIGHLAND": "Public Transportation",
    "JUNIOR HIGH SCHOOL": "Educational Institution",
    "MTA - GOLD LINE - SOTO": "Public Transportation",
    "MTA - EXPO LINE - LA CIENEGA/JEFFERSON": "Public Transportation",
    "MTA - BLUE LINE - SAN PEDRO": "Public Transportation",
    "MTA - SILVER LINE - HARBOR FWY": "Public Transportation",
    "GAS STATION": "Commercial Area",
    "MTA - GOLD LINE - SOUTHWEST MUSEUM": "Public Transportation",
    "VEHICLE STORAGE LOT (CARS, TRUCKS, RV'S, BOATS, TRAILERS, ETC.)": "Commercial Area",
    "OTHER PLACE OF WORSHIP": "Religious Institution",
    "MTA - ORANGE LINE - WOODLEY": "Public Transportation",
    "MASS GATHERING LOCATION": "Event Venue",
    "DRIVE THRU BANKING (WINDOW)*": "Dining Establishment",
    "GOVERNMENT FACILITY (FEDERAL,STATE, COUNTY & CITY)": "Government Facility",
    "GUN/SPORTING GOODS": "Commercial Area",
    "MTA - RED LINE - NORTH HOLLYWOOD": "Public Transportation",
    "MTA - RED LINE - UNION STATION": "Public Transportation",
    "FRAT HOUSE/SORORITY/DORMITORY": "Residential Area",
    "PUBLIC RESTROOM(INDOORS-INSIDE)": "Public Area",
    "BUS STOP/LAYOVER (ALSO QUERY 124)": "Public Transportation",
    "MTA - EXPO LINE - EXPO/SEPULVEDA": "Public Transportation",
    "MTA - RED LINE - HOLLYWOOD/VINE": "Public Transportation",
    "PAWN SHOP": "Commercial Area",
    "CLEANER/LAUNDROMAT": "Commercial Area",
    "POOL-PUBLIC/OUTDOOR OR INDOOR*": "Recreational Area",
    "MARKET": "Commercial Area",
    "ABANDONED BUILDING ABANDONED HOUSE": "Residential Area",
    "ENTERTAINMENT/COMEDY CLUB (OTHER)": "Entertainment Venue",
    "HARDWARE/BUILDING SUPPLY": "Commercial Area",
    "AUTO SALES LOT": "Commercial Area",
    "BUS, SCHOOL, CHURCH": "Public Transportation",
    "MTA - GOLD LINE - CHINATOWN": "Public Transportation",
    "GROUP HOME": "Residential Area",
    "MAIL BOX": "Public Area",
    "TOOL SHED*": "Outdoor Area",
    "GARAGE/CARPORT": "Residential Area",
    "DRUG STORE": "Commercial",
    "MTA - PURPLE LINE - PERSHING SQUARE": "Public Transportation",                            
    "PRIVATE SCHOOL/PRESCHOOL": "Educational Institution",                                       
    "MTA - ORANGE LINE - VAN NUYS": "Public Transportation",                                   
    "DEPARTMENT STORE": "Commercial Area",
    "MTA - RED LINE - 7TH AND METRO CENTER": "Public Transportation",
    "DIY CENTER (LOWE'S,HOME DEPOT,OSH,CONTRACTORS WAREHOUSE)": "Commercial Area",
    "PATIO*": "Outdoor Area",
    "FACTORY": "Industrial Area",
    "WEBSITE": "Online",
    "SINGLE FAMILY DWELLING": "Residential Area",
    "ALLEY": "Public Area",
    "MTA - PURPLE LINE - UNION STATION": "Public Transportation",
    "BEAUTY/BARBER SHOP": "Commercial Area",
    "PARKING LOT": "Public Area",
    "OTHER/OUTSIDE": "Unknown",
    "AUTO SUPPLY STORE*": "Commercial Area",
    "TATTOO PARLOR*": "Commercial Area",
    "CELL PHONE STORE": "Commercial Area",
    "SWAP MEET": "Commercial Area",
    "MTA - ORANGE LINE - BALBOA": "Public Transportation",
    "RECORD-CD MUSIC/COMPUTER GAME STORE": "Commercial Area",
    "DETENTION/JAIL FACILITY": "Government Facility",
    "CHURCH/CHAPEL (CHANGED 03-03 FROM CHURCH/TEMPLE)": "Religious Institution",
    "MTA - ORANGE LINE - CHATSWORTH": "Public Transportation",
    "SKATEBOARD FACILITY/SKATEBOARD PARK*": "Recreational Area",
    "PORCH, RESIDENTIAL": "Residential Area",
    "DAM/RESERVOIR": "Utility Facility",
    "POLICE FACILITY": "Government Facility",
    "LA UNION STATION (NOT LINE SPECIFIC)": "Public Transportation",
    "BUS DEPOT/TERMINAL, OTHER THAN MTA": "Public Transportation",
    "PARKING UNDERGROUND/BUILDING": "Public Area",
    "MTA - SILVER LINE - PACIFIC COAST HWY": "Public Transportation",
    "STORAGE SHED": "Outdoor Area",
    "OTHER STORE": "Commercial Area",
    "VETERINARIAN/ANIMAL HOSPITAL": "Healthcare Facility",
    "SURPLUS SURVIVAL STORE": "Commercial Area",
    "ELECTRONICS STORE (IE:RADIO SHACK, ETC.)": "Commercial Area",
    "METHADONE CLINIC": "Healthcare Facility",
    "MTA - RED LINE - CIVIC CENTER/GRAND PARK": "Public Transportation",
    "OTHER INTERSTATE, CHARTER BUS": "Public Transportation",
    "MTA - EXPO LINE - EXPO/WESTERN": "Public Transportation",
    "FINANCE COMPANY": "Financial Institution",
    "NIGHT CLUB (OPEN EVENINGS ONLY)": "Entertainment Venue",
    "LIQUOR STORE": "Commercial Area",
    "VALET": "Service",
    "MEDICAL MARIJUANA FACILITIES/BUSINESSES": "Healthcare Facility",
    "TRASH CAN/TRASH DUMPSTER": "Public Area",
    "CLOTHING STORE": "Commercial Area",
    "MTA PROPERTY OR PARKING LOT": "Public Transportation",
    "PROJECT/TENEMENT/PUBLIC HOUSING": "Residential Area",
    "BEACH": "Outdoor Area",
    "AUTO DEALERSHIP (CHEVY, FORD, BMW, MERCEDES, ETC.)": "Commercial Area",
    "MTA - GOLD LINE - LINCOLN/CYPRESS": "Public Transportation",
    "UNDERPASS/BRIDGE*": "Outdoor Area",
    "COMPUTER SERVICES/REPAIRS/SALES": "Commercial Area",
    "DISCOUNT STORE (99 CENT,DOLLAR,ETC.": "Commercial Area",
    "FIRE STATION": "Government Facility",
    "MTA - GOLD LINE - UNION STATION": "Public Transportation",
    "TRADE SCHOOL (MEDICAL-TECHNICAL-BUSINESS)*": "Educational Institution",
    "MTA - PURPLE LINE - WESTLAKE/MACARTHUR PARK": "Public Transportation",
    "MTA - EXPO LINE - WESTWOOD/RANCHO PARK": "Public Transportation",
    "MTA - RED LINE - WESTLAKE/MACARTHUR PARK": "Public Transportation",
    "BOOK STORE": "Commercial Area",
    "MTA - GOLD LINE - INDIANA": "Public Transportation",
    "BUS-CHARTER/PRIVATE": "Public Transportation",
    "TRAIN DEPOT/TERMINAL, OTHER THAN MTA": "Public Transportation",
    "CONSTRUCTION SITE": "Construction Area",
    "DAY CARE/CHILDREN*": "Day Care Facility",
    "AUTO REPAIR SHOP": "Commercial Area",
    "RECYCLING CENTER": "Utility Facility",
    "MTA - EXPO LINE - PICO": "Public Transportation",
    "OTHER PREMISE": "Unknown",
    "MTA BUS": "Public Transportation",
    "PUBLIC STORAGE": "Commercial Area",
    "PHARMACY INSIDE STORE OR SUPERMARKET*": "Commercial Area",
    "MTA - EXPO LINE - EXPO/VERMONT": "Public Transportation",
    "SAVINGS & LOAN": "Financial Institution",
    "MTA - EXPO LINE - LATTC/ORTHO INSTITUTE": "Public Transportation",
    "MTA - BLUE LINE - 7TH AND METRO CENTER": "Public Transportation",
    "TRUCK, COMMERICAL": "Transportation",
    "SHORT-TERM VACATION RENTAL": "Hospitality Venue",
    "MTA - BLUE LINE - GRAND/LATTC": "Public Transportation",
    "THE BEVERLY CONNECTION": "Commercial Area",
    "MANUFACTURING COMPANY": "Industrial Area",
    "MTA - ORANGE LINE - ROSCOE": "Public Transportation",
    "MTA - ORANGE LINE - DE SOTO": "Public Transportation",
    "MTA - ORANGE LINE - NORDHOFF": "Public Transportation",
    "MTA - GREEN LINE - AVALON": "Public Transportation",
    "MTA - ORANGE LINE - SHERMAN WAY": "Public Transportation",
    "MTA - BLUE LINE - VERNON": "Public Transportation",
    "GARMENT MANUFACTURER": "Industrial Area",
    "MTA - ORANGE LINE - VALLEY COLLEGE": "Public Transportation",
    "MTA - PURPLE LINE - WILSHIRE/VERMONT": "Public Transportation",
    "MTA - ORANGE LINE - CANOGA": "Public Transportation",
    "MUSCLE BEACH": "Recreational Area",
    "MTA - PURPLE LINE - WILSHIRE/NORMANDIE": "Public Transportation",
    "SKATING RINK*": "Recreational Area",
    "VIDEO RENTAL STORE": "Commercial Area",
    "MTA - BLUE LINE - 103RD/WATTS TOWERS": "Public Transportation",
    "MTA - GREEN LINE - HARBOR FWY": "Public Transportation",
    "MTA - PURPLE LINE - WILSHIRE/WESTERN": "Public Transportation",
    "BANK DROP BOX/MONEY DROP-OUTSIDE OF BANK*": "Financial Institution",
    "RETIRED (DUPLICATE) DO NOT USE THIS CODE": "Unknown",
    "DEPT OF DEFENSE FACILITY": "Government Facility",
    "MTA - SILVER LINE - LAC/USC MEDICAL CENTER": "Public Transportation",
    "TRAM/STREETCAR(BOXLIKE WAG ON RAILS)*": "Public Transportation",
    "CHEMICAL STORAGE/MANUFACTURING PLANT": "Industrial Area",
    "TRAIN, OTHER THAN MTA (ALSO QUERY 809/810/811)": "Public Transportation",
    "HOCKEY RINK/ICE HOCKEY": "Recreational Area",
    "MTA - SILVER LINE - MANCHESTER": "Public Transportation",
    "HARBOR FRWY STATION (NOT LINE SPECIFIC)": "Public Transportation",
    "BEAUTY SUPPLY STORE": "Commercial Area",
    "CATERING/ICE CREAM TRUCK": "Mobile Vendor",
    "COFFEE SHOP (STARBUCKS, COFFEE BEAN, PEET'S, ETC.)": "Dining Establishment",
    "FREEWAY": "Transportation",
    "HANDBALL COURTS": "Recreational Area",
    "HEALTH SPA/GYM": "Recreational Area",
    "HIGH-RISE BUILDING": "Residential Area",
    "HORSE RACING/SANTA ANITA PARK*": "Recreational Area",
    "MEDICAL/DENTAL OFFICES": "Healthcare Facility",
    "MORTUARY": "Healthcare Facility",
    "MTA - BLUE LINE - WASHINGTON": "Public Transportation",
    "MTA - EXPO LINE - JEFFERSON/USC": "Public Transportation",
    "MTA - EXPO LINE - PALMS": "Public Transportation",
    "MTA - GOLD LINE - HERITAGE SQ": "Public Transportation",
    "MTA - ORANGE LINE - LAUREL CANYON": "Public Transportation",
    "MTA - ORANGE LINE - NORTH HOLLYWOOD": "Public Transportation",
    "MTA - ORANGE LINE - PIERCE COLLEGE": "Public Transportation",
    "MTA - ORANGE LINE - WOODMAN": "Public Transportation",
    "MTA - RED LINE - UNIVERSAL CITY/STUDIO CITY": "Public Transportation",
    "MTA - RED LINE - WILSHIRE/VERMONT": "Public Transportation",
    "MTA - SILVER LINE - SLAUSON": "Public Transportation",
    "OTHER RESIDENCE": "Residential Area",
    "SHOPPING MALL (COMMON AREA)": "Commercial Area",
    "SPORTS VENUE, OTHER": "Sports Venue",
    "TERMINAL, OTHER THAN MTA": "Public Transportation",
    "TUNNEL": "Transportation",
    "VEHICLE, PASSENGER/TRUCK": "Transportation",
    "VISION CARE FACILITY*": "Healthcare Facility",
    "WATER FACILITY": "Utility Facility",
    "Waterfront Area": "Waterfront Area"
}


# COMMAND ----------

df_transformed = df_transformed.withColumn("Premis Desc", col("Premis Desc").cast("string"))  
for key, value in premise_mapping.items():
    df_transformed = df_transformed.replace(key, value, "Premis Desc")

# COMMAND ----------

  repl={
  "BALCONY*": "Outdoor Area",
  "BANKING INSIDE MARKET-STORE *": "Financial Institution",
  "MOSQUE*": "Religious Institution",
  "VISION CARE FACILITY*": "Healthcare Facility",
  "BEAUTY SUPPLY STORE": "Commercial Area"
  }

# COMMAND ----------

df_transformed = df_transformed.withColumn("Premis Desc", col("Premis Desc").cast("string"))  
for key, value in repl.items():
    df_transformed = df_transformed.replace(key, value, "Premis Desc")

# COMMAND ----------

weapon_mapping = {
    "BELT FLAILING INSTRUMENT/CHAIN": "Belt/Chain",
    "SEMI-AUTOMATIC PISTOL": "Pistol",
    "OTHER FIREARM": "Firearm",
    "BOTTLE": "Bottle",
    "OTHER KNIFE": "Knife",
    "SWITCH BLADE": "Switch Blade",
    "null": "Unknown",
    "BOWIE KNIFE": "Knife",
    "SAWED OFF RIFLE/SHOTGUN": "Shotgun",
    "AIR PISTOL/REVOLVER/RIFLE/BB GUN": "Pistol",
    "OTHER CUTTING INSTRUMENT": "Other",
    "ASSAULT WEAPON/UZI/AK47/ETC": "Assault Rifle",
    "UNKNOWN FIREARM": "Firearm",
    "AUTOMATIC WEAPON/SUB-MACHINE GUN": "Assault Rifle",
    "STICK": "Stick",
    "SWORD": "Sword",
    "BOMB THREAT": "Bomb",
    "TOY GUN": "Toy Gun",
    "SCREWDRIVER": "Screwdriver",
    "VERBAL THREAT": "Verbal Threats",
    "FIXED OBJECT": "Fixed Object",
    "MACHETE": "Knife",
    "BOW AND ARROW": "Bow & Arrow",
    "M1-1 SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
    "EXPLOXIVE DEVICE": "Other",
    "SEMI-AUTOMATIC RIFLE": "Semi-Automatic Rifle",
    "MAC-10 SEMIAUTOMATIC ASSAULT WEAPON": "Semi-Automatic Rifle",
    "HAMMER": "Hammer",
    "TIRE IRON": "Tire Iron",
    "FOLDING KNIFE": "Knife",
    "REVOLVER": "Firearm",
    "ROCK/THROWN OBJECT": "Projectile/Object",
    "KNIFE WITH BLADE OVER 6 INCHES IN LENGTH": "Knife",
    "ROPE/LIGATURE": "Rope",
    "ANTIQUE FIREARM": "Firearm",
    "RAZOR": "Knife",
    "VEHICLE": "Vehicle",
    "PIPE/METAL PIPE": "Pipe",
    "STARTER PISTOL/REVOLVER": "Pistol",
    "CLUB/BAT": "bat",
    "STRAIGHT RAZOR": "Knife",
    "UNKNOWN TYPE CUTTING INSTRUMENT": "Knife",
    "MACE/PEPPER SPRAY": "Pepper Spray",
    "LIQUOR/DRUGS": "Drugs",
    "MARTIAL ARTS WEAPONS": "Martial Arts Tool",
    "DOG/ANIMAL (SIC ANIMAL ON)": "Animal",
    "CLEAVER": "Knife",
    "SCALDING LIQUID": "Chemical",
    "STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)": "Physical",
    "DEMAND NOTE": "Demand Note",
    "KNIFE WITH BLADE 6INCHES OR LESS": "Knife",
    "BRASS KNUCKLES": "Brass Knuckles",
    "AXE": "Axe",
    "SYRINGE": "Syringe",
    "CAUSTIC CHEMICAL/POISON": "Chemical",
    "BLUNT INSTRUMENT": "Object",
    "RIFLE": "Assault Rifle",
    "STUN GUN": "Stun gun",
    "KITCHEN KNIFE": "Knife",
    "FIRE": "Fire",
    "DIRK/DAGGER": "Knife",
    "SCISSORS": "Scissors",
    "UNKNOWN WEAPON/OTHER WEAPON": "Unknown",
    "SHOTGUN": "Shotgun",
    "SIMULATED GUN": "Other",
    "PHYSICAL PRESENCE": "Other",
    "GLASS": "Object",
    "MAC-11 SEMIAUTOMATIC ASSAULT WEAPON": "Semi-Automatic Rifle",
    "RAZOR BLADE": "Knife",
    "HAND GUN": "Firearm",
    "CONCRETE BLOCK/BRICK": "Object",
    "BOARD": "Object",
    "ICE PICK": "Object",
    "HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
    "BLACKJACK": "Other",
    "UZI SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
    "HECKLER & KOCH 91 SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
    "RELIC FIREARM": "Firearm",
    "UNK TYPE SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
    "M-14 SEMIAUTOMATIC ASSAULT RIFLE": "Semi-Automatic Rifle",
}

# COMMAND ----------

df_transformed = df_transformed.withColumn("Weapon Desc", col("Weapon Desc").cast("string"))  
for key, value in weapon_mapping.items():
    df_transformed = df_transformed.replace(key, value, "Weapon Desc")

# COMMAND ----------

# MAGIC %md
# MAGIC Date Rptd is usesless.
# MAGIC futher data occ needs to be dissected to month and day as year is again a useless element -> for association
# MAGIC

# COMMAND ----------

df_transformed=df_transformed.drop("Date Rptd")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Converting "Date OCC" to timestamp and extract month and day

# COMMAND ----------

df_transformed = df_transformed.withColumn("Timestamp", unix_timestamp("Date OCC", "MM-dd-yyyy").cast("timestamp"))

# Extract the month name and day of the week
df_transformed = df_transformed.withColumn("Month", date_format("Timestamp", "MMMM"))
df_transformed = df_transformed.withColumn("Day", date_format("Timestamp", "EEEE"))
df_transformed=df_transformed.drop("Timestamp")

# COMMAND ----------

# MAGIC %md 
# MAGIC Filter rows with specific conditions for each column. Where we removed values for Vict Age that included -2,-1 or 0 as that is not possible. Null values were also deleted from the other columns to ensure data compelteness.
# MAGIC

# COMMAND ----------

df_transformed = df_transformed.filter(~col("Vict Age").isin(-2, -1, 0))
df_transformed = df_transformed.filter(~col("Vict Sex").isin("null", "H"))
df_transformed = df_transformed.filter(~col("Vict Descent").isin("null"))
df_transformed = df_transformed.filter(~col("Premis Desc").isin("null"))
df_transformed = df_transformed.filter(~col("Weapon Desc").isin("null"))
df_transformed = df_transformed.filter(~col("Premis Desc").isin("null"))