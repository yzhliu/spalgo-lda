package org.apache.spark.mllib.spalgo.lda

class LikelihoodAggregator {
  // g.vertices.filter(t => t._1 > 0).map(_._2).aggregate(new VD(numTopics))(_ + _, _ + _)
  var likWordsGivenTopics: Double = 0d
  var likTopics: Double = 0d
}
