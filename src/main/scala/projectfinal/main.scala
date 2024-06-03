package projectfinal

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/Users/Jinrong/OneDrive/Desktop/CSC369")
    val conf = new SparkConf().setAppName("asgn5").setMaster("local[*]")
    val sc = new SparkContext(conf)

    def calculateMean(values: Array[String]): Double = {
      val numericValues = values.filter(_.nonEmpty).map(_.toDouble)
      if (numericValues.isEmpty) 0.0 else numericValues.sum / numericValues.length
    }

    val expenditures = sc.textFile("expenditures.csv")
      .map(line => line.split(","))
      .map(columns => (columns(0), columns.tail))
      .map { case (c, l) => (c, calculateMean(l)) }

    val completion = sc.textFile("completion.csv")
      .map(line => line.split(",", 54))
      .map(columns => {
        val country = columns(0)
        val values = columns.tail
        val filteredValues = values.slice(25, values.length - 4)
        (country, filteredValues)
      })
      .map { case (c, l) => (c, calculateMean(l)) }

    val join = expenditures.join(completion)
    join.foreach(println)

    join.sortBy(_._2._1, ascending = false).take(10).foreach(println(_))

    join.sortBy(_._2._2, ascending = false).take(10).foreach(println(_))

    val spark = SparkSession.builder()
      .appName("Linear Regression Example")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val data = join.map { case (_, (exp, compl)) =>
      (exp.toDouble, compl.toDouble)
    }.toDF("expenditure", "completion")

    val assembler = new VectorAssembler()
      .setInputCols(Array("expenditure"))
      .setOutputCol("features")

    val assembledData = assembler.transform(data)

    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2))

    val lr = new LinearRegression()
      .setLabelCol("completion")
      .setFeaturesCol("features")

    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

    val predictions = lrModel.transform(testData)

    predictions.select("expenditure", "completion", "prediction").show()

    val evaluator = new org.apache.spark.ml.evaluation.RegressionEvaluator()
      .setLabelCol("completion")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data: $rmse")

  }
}
