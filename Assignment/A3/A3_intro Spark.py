from pyspark import SparkContext
sc = SparkContext(master="local[4]")

# oneFile = sc.textFile('s3://risamyersbucket/A3/T201607PDPI+BNFT.csv')
allFiles = sc.textFile('s3a://risamyersbucket/A3/')


def typeCheck(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


data = allFiles.filter(lambda x: typeCheck(str(x.split(',')[6])) == True)

element = data.map(lambda x: (x.split(',')[9], x.split(',')[6]))
# element.collect()
res = element.reduceByKey(lambda x, y: float(x) + float(y))

res = res.sortByKey()

res.coalesce(1).saveAsTextFile("output_task_1")

data = allFiles.filter(lambda x: typeCheck(str(x.split(',')[6])) == True)
element_task2 = data.map(lambda x: (
    x.split(',')[2], float(x.split(',')[6])))

res2 = element_task2.reduceByKey(lambda x, y: x + y)

# res2.top(5, key=lambda x: x[1])

res_rdd = sc.parallelize(res2.top(5, key=lambda x: x[1]))

res_rdd.coalesce(1).saveAsTextFile("output_task_2")
