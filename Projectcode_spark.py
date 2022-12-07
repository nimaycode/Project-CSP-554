from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("sparktest").getOrCreate()
spark

df = spark.read.csv("/Users/nimay/Desktop/CSP-554/Project/Heartdiseasedataset.csv", inferSchema=True, header=True)
df.show(2)
df.count()
df.printSchema()
df.groupBy('target').count().show()

from pyspark.sql.functions import *

def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if(nullRows > 0):
            temp = k,nullRows,(nullRows/numRows)*100
            null_columns_counts.append(temp)
    return(null_columns_counts)

null_columns_calc_list = null_value_calc(df)
if null_columns_calc_list : 
    spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count','Null_Value_Percent']).show()
else :
    print("Data is clean with no null values")

from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.sql.types import * 

from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.sql.functions import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def MLClassifierDFPrep(df,input_columns,dependent_var,treat_outliers=True,treat_neg_values=True):
    renamed = df.withColumn("label_str", df[dependent_var].cast(StringType()))
    indexer = StringIndexer(inputCol="label_str", outputCol="label")
    indexed = indexer.fit(renamed).transform(renamed)
    print(indexed.groupBy(dependent_var,"label").count().show(100))

numeric_inputs = []
string_inputs = []
for column in input_columns:
    if str(indexed.schema[column].dataType) == 'StringType':
    indexer = StringIndexer(inputCol=column, outputCol=column+"_num") 
    indexed = indexer.fit(indexed).transform(indexed)
    new_col_name = column+"_num"
    string_inputs.append(new_col_name)
    else:
    numeric_inputs.append(column)
            
if treat_outliers == True:
    print("normality achieved!")
    d = {}
    for col in numeric_inputs: 
        d[col] = indexed.approxQuantile(col,[0.01,0.99],0.25)
    for col in numeric_inputs:
        skew = indexed.agg(skewness(indexed[col])).collect()
        skew = skew[0][0]
        if skew > 1:
            indexed = indexed.withColumn(col, \
            log(when(df[col] < d[col][0],d[col][0])\
            .when(indexed[col] > d[col][1], d[col][1])\
            .otherwise(indexed[col] ) +1).alias(col))
            print(col+" positive skewness. (skew =)",skew,")")
        else if skew < -1:
            indexed = indexed.withColumn(col, \
            exp(when(df[col] < d[col][0],d[col][0])\
            .when(indexed[col] > d[col][1], d[col][1])\
            .otherwise(indexed[col] )).alias(col))
            print(col+"negative skewness. (skew =",skew,")")

minimums = df.select([min(c).alias(c) for c in df.columns if c in numeric_inputs])
min_array = minimums.select(array(numeric_inputs).alias("mins"))
df_minimum = min_array.select(array_min(min_array.mins)).collect()
df_minimum = df_minimum[0][0]

features_list = numeric_inputs + string_inputs
assembler = VectorAssembler(inputCols=features_list,outputCol='features')
output = assembler.transform(indexed).select('features','label')

if df_minimum < 0:
    print(" ")
    print("WARNING: The Naive Bayes negetive values")
    print(" ")

if treat_neg_values == True:
    print("correct")
    print(" ")
    print("rescaling")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(output)

scaled_data = scalerModel.transform(output)
final_data = scaled_data.select('label','scaledFeatures')
final_data = final_data.withColumnRenamed('scaledFeatures','features')
print("Done!")

else:
    print("correctr")
    print("dataframe unscaled.")
    final_data = output

return final_data

def ClassTrainEval(classifier,features,classes,folds,train,test):

    def FindMtype(classifier):
    M = classifier
    Mtype = type(M).__name__

return Mtype

Mtype = FindMtype(classifier)


def IntanceFitModel(Mtype,classifier,classes,features,folds,train):

if Mtype in("LinearSVC","GBTClassifier") and classes != 2:
    print(Mtype," binary classification error")
    return
    if Mtype in("NaiveBayes","RandomForestClassifier","LinearSVC","DecisionTreeClassifier"):

if Mtype in("NaiveBayes"):
    paramGrid = (ParamGridBuilder() \
    .addGrid(classifier.smoothing, [0.0, 0.2, 0.4, 0.6]) \
    .build())

if Mtype in("RandomForestClassifier"):
    paramGrid = (ParamGridBuilder() \
    .addGrid(classifier.maxDepth, [2, 5, 10])
    .build())

if Mtype in("LinearSVC"):
    paramGrid = (ParamGridBuilder() \
    .addGrid(classifier.maxIter, [10, 15]) \
    .addGrid(classifier.regParam, [0.1, 0.01]) \
    .build())

if Mtype in("DecisionTreeClassifier"):
    paramGrid = (ParamGridBuilder() \
    .addGrid(classifier.maxBins, [10, 20, 40, 80, 100]) \
    .build())

crossval = CrossValidator(estimator=classifier,
estimatorParamMaps=paramGrid,
evaluator=MulticlassClassificationEvaluator(),
numFolds=folds)
fitModel = crossval.fit(train)
return fitModel

fitModel = IntanceFitModel(Mtype,classifier,classes,features,folds,train)

if fitModel is not None:

    if Mtype in("DecisionTreeClassifier","RandomForestClassifier"):
        BestModel = fitModel.bestModel
        print(" ")
        print('\033[1m' + Mtype," Top 20"+ '\033[0m')
        print("(add to 1)")
        print("Lowest score")
        print(" ")
        featureImportances = BestModel.featureImportances.toArray()
        imp_scores = []
    for x in featureImportances:
        imp_scores.append(float(x))
        result = spark.createDataFrame(zip(input_columns,imp_scores), schema=['feature','score'])
        print(result.orderBy(result["score"].desc()).show(truncate=False))

if Mtype in("DecisionTreeClassifier"):
    global DT_featureimportances
    DT_featureimportances = BestModel.featureImportances.toArray()
    global DT_BestModel
    DT_BestModel = BestModel
if Mtype in("RandomForestClassifier"):
    global RF_featureimportances
    RF_featureimportances = BestModel.featureImportances.toArray()
    global RF_BestModel
    RF_BestModel = BestModel

if Mtype in("LinearSVC"):
    BestModel = fitModel.bestModel
    print(" ")
    print('\033[1m' + Mtype + '\033[0m')
    print("Intercept: " + str(BestModel.intercept))
    print('\033[1m' + "Top 20"+ '\033[0m')
    print("relative score")
    coeff_array = BestModel.coefficients.toArray()
    coeff_scores = []
for x in coeff_array:
    coeff_scores.append(float(x))
    result = spark.createDataFrame(zip(input_columns,coeff_scores), schema=['feature','coeff'])
    print(result.orderBy(result["coeff"].desc()).show(truncate=False))
    global LSVC_coefficients
    LSVC_coefficients = BestModel.coefficients.toArray()
    global LSVC_BestModel
    LSVC_BestModel = BestModel

columns = ['Classifier', 'Result']

if Mtype in("LinearSVC") and classes != 2:
    Mtype = [Mtype]
    score = ["N/A"]
    result = spark.createDataFrame(zip(Mtype,score), schema=columns)
else:
    predictions = fitModel.transform(test)
    MC_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = (MC_evaluator.evaluate(predictions))*100
    Mtype = [Mtype]
    score = [str(accuracy)]
    result = spark.createDataFrame(zip(Mtype,score), schema=columns)
    result = result.withColumn('Result',result.Result.substr(0, 5))

return result

input_columns = df.columns
input_columns = input_columns[:-1]
dependent_var = 'target'
class_count = df.select(countDistinct("target")).collect()
classes = class_count[0][0]

test1_data = MLClassifierDFPrep(df,input_columns,dependent_var)
test1_data.limit(5).toPandas()

classifiers = [
LinearSVC()
,NaiveBayes()
,RandomForestClassifier()
,DecisionTreeClassifier()
] 

train,test = test1_data.randomSplit([0.7,0.3])
features = test1_data.select(['features']).collect()
folds = 3

columns = ['Classifier', 'Result']
vals = [("Place Holder","N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
new_result = ClassTrainEval(classifier,features,classes,folds,train,test)
results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")
print("!!!!!Final Results!!!!!!!!")
results.show(100,False)