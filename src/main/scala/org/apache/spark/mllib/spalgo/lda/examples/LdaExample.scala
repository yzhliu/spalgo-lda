package org.apache.spark.mllib.spalgo.lda.examples

import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.spalgo.lda.CgsLda
import org.apache.spark.{SparkConf, SparkContext}

object LdaExample {
  def main (args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass.getSimpleName)
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("data/sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(x => x.toDouble)))
    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    if (args(0) == "gibbs") {
      // Cluster the documents into three topics using LDA
      val cgsLda = new CgsLda(10.0, 0.01, 3)
      cgsLda.train(corpus, 30)
      println("Learned topics (as distributions over vocab of " + cgsLda.numWords + " words):")
      val topics = cgsLda.estimateTopicDistributions()
      for (topic <- Range(0, 3)) {
        print("Topic " + topic + ":")
        for (word <- Range(0, cgsLda.numWords)) {
          print(" " + topics(word, topic))
        }
        println()
      }
    } else if (args(0) == "em") {
      // Cluster the documents into three topics using LDA
      val ldaModel = new LDA().setK(3).run(corpus)
      val distLdaModel = ldaModel.asInstanceOf[DistributedLDAModel]
      println(s"log-likelihood: ${distLdaModel.logLikelihood}")
      // Output topics. Each is a distribution over words (matching word count vectors)
      println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
      val topics = ldaModel.topicsMatrix
      for (topic <- Range(0, 3)) {
        print("Topic " + topic + ":")
        for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
        println()
      }
    }
  }
}
