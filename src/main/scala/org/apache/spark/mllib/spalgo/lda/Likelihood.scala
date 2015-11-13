package org.apache.spark.mllib.spalgo.lda

import org.apache.spark.graphx.VertexId
import org.apache.spark.mllib.spalgo.lda.CgsLda

class LikelihoodAggregator(val numTopics: Int) {
  // g.vertices.filter(t => t._1 > 0).map(_._2).aggregate(new VD(numTopics))(_ + _, _ + _)
  var likWordsGivenTopics: Double = 0d
  var likTopics: Double = 0d
  // TODO
  val globalTopicCount: VertexData = _
  val numWords: Int = _
  val numDocs: Int = _

  def map(vertexId: VertexId, data: VertexData) = {
    // using boost::math::lgamma;
    require(data.numTopics == numTopics)
    val aggregator = new LikelihoodAggregator(numTopics)
    if (vertexId > 0) { // is word
      for (t <- 0 until numTopics) {
        val topicCount = data(t)
        require(topicCount >= 0)
        // TODO
        aggregator.likWordsGivenTopics += BETA_LGAMMA(topicCount)
      }
    } else { // is doc
      var ntokensInDoc = 0d
      for(t <- 0 until numTopics) {
        val topicCount = data(t)
        require(topicCount >= 0)
        // TODO
        aggregator.likTopics += ALPHA_LGAMMA(topicCount)
        ntokensInDoc += topicCount
      }
      aggregator.likTopics -= lgamma(ntokensInDoc + numTopics * ALPHA)
    }
    aggregator
  }

  def finalize(agg1: LikelihoodAggregator, agg2: LikelihoodAggregator) {
    // Address the global sum terms
    var denominator = 0d
    for (t <- 0 until numTopics) {
      val value = globalTopicCount(t)
      require(value >= 0)
      denominator += lgamma(value + numWords * BETA);
    }

    val likWordsGivenTopics =
      numTopics * (lgamma(numWords * BETA) - numWords * lgamma(BETA)) -
        denominator + total.likWordsGivenTopics

    val likTopics =
      numDocs * (lgamma(numTopics * ALPHA) - numTopics * lgamma(ALPHA)) +
        total.likTopics

    likWordsGivenTopics + likTopics
  }
}
