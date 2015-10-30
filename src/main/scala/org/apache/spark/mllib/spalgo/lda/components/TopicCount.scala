package org.apache.spark.mllib.spalgo.lda.components

class TopicCount(numTopics: Int) extends Serializable {
  val topics = Array.ofDim[Int](numTopics)

  def add(otherTopicCount: TopicCount) = {
    require(otherTopicCount.topics.length == topics.length)
    for (i <- 0 until topics.length) {
      topics(i) += otherTopicCount.topics(i)
    }
    this
  }
}
