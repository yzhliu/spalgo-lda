package org.apache.spark.mllib.spalgo.lda

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BreezeDenseVector, Matrix => BM, SparseVector => BreezeSparseVector, Vector => BreezeVector}
import breeze.numerics.lgamma
import breeze.stats.distributions.Multinomial
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer
import scala.util.Random

class EdgeData(val dimension: Int, val numTopics: Int, rnd: Random) extends Serializable {
  var topicAssignments = Array.ofDim[Int](dimension)
  for (i <- 0 until dimension) {
    topicAssignments(i) = rnd.nextInt(numTopics)
  }

  // FIXME
  def this(other: EdgeData) {
    this(other.dimension, other.numTopics, new Random())
    topicAssignments = other.topicAssignments.clone()
  }

  override def clone() = {
    new EdgeData(this)
  }

  def size = topicAssignments.length

  def apply(idx: Int) = topicAssignments(idx)

  override def toString = {
    s"TopicAssignments[${topicAssignments.mkString(",")}]"
  }
}

class VertexData(val numTopics: Int) extends Serializable {
  var topicAttachedCounts = BreezeVector.zeros[Int](numTopics)

  def this(other: VertexData) {
    this(other.numTopics)
    topicAttachedCounts = BreezeVector(other.topicAttachedCounts.toArray.clone())
  }

  override def clone() = {
    new VertexData(this)
  }

  def increment(topicId: Int): Unit = {
    increment(topicId, 1)
  }
  def increment(topicId: Int, delta: Int): Unit = {
    require(topicId >= 0 && topicId < topicAttachedCounts.length)
    topicAttachedCounts(topicId) += delta
  }
  def apply(topicId: Int): Int = {
    require(topicId >= 0 && topicId < topicAttachedCounts.length)
    topicAttachedCounts(topicId)
  }
  def add(otherVertexData: VertexData): VertexData = {
    require(topicAttachedCounts.size == otherVertexData.topicAttachedCounts.size)
    val newTopicAttachedCounts = topicAttachedCounts + otherVertexData.topicAttachedCounts
    val newData = new VertexData(numTopics)
    newData.topicAttachedCounts = newTopicAttachedCounts
    newData
  }
  def +(other: VertexData): VertexData = {
    add(other)
  }
  override def toString = {
    // TODO
    s"TopicCounts[${topicAttachedCounts.toArray.mkString(",")}]"
  }
}

class CgsLda(val alpha: Double, val beta: Double, val numTopics: Int) extends Serializable {
  private type VD = VertexData
  private type ED = EdgeData
  // edge assignments
  private type Msg = VertexData

  var numWords: Int = _
  var numDocs: Int = _
  private var globalTopicCount: VertexData = _
  private var graph: Graph[VD, ED] = _

  // vertex-data: (should be atomic) count of appearance of each topic
  // edge-data: the topic assigned to every appearance of the word in this document
  def load(documents: RDD[(Long, Vector)]): Graph[VD, ED] = {
    this.numWords = documents.first()._2.size
    this.numDocs = documents.count().asInstanceOf[Int]
    val edges = documents.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    graph = Graph.fromEdges(edges, null, edgeStorageLevel = StorageLevel.MEMORY_AND_DISK, vertexStorageLevel = StorageLevel.MEMORY_AND_DISK)
    graph
  }

  def train(documents: RDD[(Long, Vector)], maxIterations: Int) = {
    load(documents)
    var i = 0
    while (i < maxIterations) {
      nextIteration()
      i += 1
    }
    // TODO: unpersist graph
  }

  def estimateTopicDistributions(): Matrix = {
    val messages: VertexRDD[Msg]
      = graph.aggregateMessages(sendMessage, mergeMessage)//.persist(StorageLevel.MEMORY_AND_DISK)

    val newG = graph.joinVertices(messages) {
      case (vertexId, null, newData) => newData
    }
    // calculate topic counts for words and documents
    globalTopicCount = calculateGlobalTopicCount(newG)
    val wordTopicCount = newG.vertices.filter(t => t._1 > 0).aggregateByKey(new VD(numTopics))(_ + _, _ + _).collect()
    val wordTopicDistribution = DenseMatrix.zeros(numWords, numTopics)
    val betaSum = beta * numWords
    wordTopicCount.foreach { case (wordId, topicCount) =>
      require(topicCount.numTopics == numTopics)
      for (t <- 0 until numTopics) {
        val phi = (topicCount(t) + beta) / (globalTopicCount(t) + betaSum)
        wordTopicDistribution(wordId.toInt - 1, t) = phi
      }
    }
    wordTopicDistribution
  }

  private def nextIteration(): Unit = {
    val prevG = graph
    // init globalTopicCount
    val messages: VertexRDD[Msg]
      = prevG.aggregateMessages(sendMessage, mergeMessage) // TODO .persist(StorageLevel.MEMORY_AND_DISK)
    //println(s"Messages: ${messages.collectAsMap()}")
    var newG = prevG.joinVertices(messages) {
      case (vertexId, null, newData) => newData.clone()
    }
    // calculate topic counts for words and documents
    globalTopicCount = calculateGlobalTopicCount(newG)
    val logLik = calculateLogLikelihood(newG)
    println(s"********* LogLikelihood: $logLik")
    newG = newG.mapTriplets((pid: PartitionID, iter: Iterator[EdgeTriplet[VD, ED]]) => {
      // sample the topic assignments z_{di}
      iter.map(triplet => {
        val wordTopicCount = triplet.srcAttr.clone()
        val docTopicCount = triplet.dstAttr.clone()
        //require(docTopicCount.length == numTopics)
        //require(wordTopicCount.length == numTopics)
        // run the actual gibbs sampling
        val prob = BreezeVector.zeros[Double](numTopics)
        val assignment = triplet.attr
        val newAssignment = assignment.clone()
        //println(s"DocId: ${triplet.dstId}, WordId: ${triplet.srcId}")
        //println(s"Assignment: $assignment\nwordTopicCount: $wordTopicCount\ndocTopicCount: $docTopicCount")
        val betaSum = beta * numWords
        for (i <- 0 until assignment.topicAssignments.length) {
          val oldTopicId = assignment.topicAssignments(i)
          docTopicCount.increment(oldTopicId, -1)
          wordTopicCount.increment(oldTopicId, -1)
          globalTopicCount.increment(oldTopicId, -1)
          for (t <- 0 until numTopics) {
            prob(t) = (alpha + docTopicCount(t)) * (beta + wordTopicCount(t)) / (betaSum + globalTopicCount(t))
            if (prob(t) < 0) {
              throw new IllegalArgumentException(s"t=$t, docTopic=${docTopicCount(t)}, " +
                s"wordTopic=${wordTopicCount(t)}, globalTopic=${globalTopicCount(t)}")
            }
          }
          val rnd = Multinomial(prob)
          val newTopicId = rnd.draw()
          newAssignment.topicAssignments(i) = newTopicId
          docTopicCount.increment(newTopicId)
          wordTopicCount.increment(newTopicId)
          globalTopicCount.increment(newTopicId)
        } // End of loop over each token
        newAssignment
      })
    }, TripletFields.All).cache()
    graph = GraphImpl(newG.vertices.mapValues(t => null), newG.edges)
    // FIXME: cause java.lang.UnsupportedOperationException:
    // Cannot change storage level of an RDD after it was already assigned a level
    //graph.persist(StorageLevel.MEMORY_AND_DISK)

    // TODO messages.unpersist(blocking = false)
    prevG.edges.unpersist(blocking = false)
    newG.vertices.unpersist(blocking = false)
    //TODO: not sure: newG.unpersistVertices(blocking = false)
  }

  private def sendMessage(triplet: EdgeContext[VD, ED, Msg]): Unit = {
    val edgeData = triplet.attr
    val message = new Msg(numTopics)
    for (w <- 0 until edgeData.size) {
      message.increment(edgeData(w))
    }
    //println(s"[SendMessage] wordId: ${triplet.srcId}, docId: ${triplet.dstId}, edgeData: ${edgeData}, message: $message")
    triplet.sendToSrc(message.clone())
    triplet.sendToDst(message.clone())
  }

  private def mergeMessage(msgA: Msg, msgB: Msg): Msg = {
    msgA + msgB
  }

  private def calculateGlobalTopicCount(g: Graph[VD, ED]): VD = {
    // words' vertexId > 0
    g.vertices.filter(t => t._1 > 0).map(_._2).aggregate(new VD(numTopics))(_ + _, _ + _)
  }

  private def calculateLogLikelihood(g: Graph[VD, ED]): Double = {
    val likAgg = g.vertices.map {
      case (vertexId: VertexId, data: VertexData) => likMap(vertexId, data)
    }.aggregate(new LikelihoodAggregator)(likReduce, likReduce)
    likFinalize(likAgg)
  }

  private def initializeEdges(
      rnd: Random,
      words: Vector,
      docId: Long,
      numTopics: Int) = {
    require(docId >= 0, s"Invalid document id $docId")
    val vertexDocId = -(docId + 1L)
    val edges = ListBuffer[Edge[ED]]()
    words.foreachActive {
      case (wordId, counter) if counter > 0.0 =>
        val vertexWordId = wordId + 1L
        val edgeData = new ED(counter.toInt, numTopics, rnd)
        edges += Edge(vertexWordId, vertexDocId, edgeData)
      case _ =>
    }
    edges
  }

  // scalastyle:off
  /**
   * From GraphLab: cgs_lda.cpp
   * \brief The Likelihood aggregators maintains the current estimate of
   * the log-likelihood of the current token assignments.
   *
   *  llik_words_given_topics = ...
   *    ntopics * (gammaln(nwords * beta) - nwords * gammaln(beta)) - ...
   *    sum_t(gammaln( n_t + nwords * beta)) +
   *    sum_w(sum_t(gammaln(n_wt + beta)));
   *
   *  llik_topics = ...
   *    ndocs * (gammaln(ntopics * alpha) - ntopics * gammaln(alpha)) + ...
   *    sum_d(sum_t(gammaln(n_td + alpha)) - gammaln(sum_t(n_td) + ntopics * alpha));
   *
   * Latex formulation:
   *
  \mathcal{L}( w | z) & = T * \left( \log\Gamma(W * \beta) - W * \log\Gamma(\beta) \right) + \\
    & \sum_{t} \left( \left(\sum_{w} \log\Gamma(N_{wt} + \beta)\right) -
           \log\Gamma\left( W * \beta + \sum_{w} N_{wt}  \right) \right) \\
    & = T * \left( \log\Gamma(W * \beta) - W * \log\Gamma(\beta) \right) -
        \sum_{t} \log\Gamma\left( W * \beta + N_{t}  \right) + \\
    & \sum_{w} \sum_{t} \log\Gamma(N_{wt} + \beta)   \\
    \\
    \mathcal{L}(z) & = D * \left(\log\Gamma(T * \alpha) - T * \log\Gamma(\alpha) \right) + \\
    & \sum_{d} \left( \left(\sum_{t}\log\Gamma(N_{td} + \alpha)\right) -
        \log\Gamma\left( T * \alpha + \sum_{t} N_{td} \right) \right) \\
    \\
    \mathcal{L}(w,z) & = \mathcal{L}(w | z) + \mathcal{L}(z)
   *
   */
  // scalastyle:on
  def likMap(vertexId: VertexId, data: VertexData) = {
    require(data.numTopics == numTopics)
    val aggregator = new LikelihoodAggregator
    if (vertexId > 0) { // is word
      for (t <- 0 until numTopics) {
        val topicCount = data(t)
        require(topicCount >= 0)
        // TODO
        aggregator.likWordsGivenTopics += lgamma(topicCount + beta)
        //aggregator.likWordsGivenTopics += BETA_LGAMMA(topicCount)
      }
    } else { // is doc
    var ntokensInDoc = 0d
      for(t <- 0 until numTopics) {
        val topicCount = data(t)
        require(topicCount >= 0)
        // TODO
        aggregator.likTopics += lgamma(topicCount + alpha)
        //aggregator.likTopics += ALPHA_LGAMMA(topicCount)
        ntokensInDoc += topicCount
      }
      aggregator.likTopics -= lgamma(ntokensInDoc + numTopics * alpha)
    }
    aggregator
  }

  def likReduce(agg1: LikelihoodAggregator, agg2: LikelihoodAggregator): LikelihoodAggregator = {
    val likAggregator = new LikelihoodAggregator()
    likAggregator.likTopics = agg1.likTopics + agg2.likTopics
    likAggregator.likWordsGivenTopics = agg1.likWordsGivenTopics + agg2.likWordsGivenTopics
    likAggregator
  }

  def likFinalize(likAggregator: LikelihoodAggregator): Double = {
    // Address the global sum terms
    var denominator = 0d
    for (t <- 0 until numTopics) {
      val value = globalTopicCount(t)
      require(value >= 0)
      denominator += lgamma(value + numWords * beta)
    }

    val likWordsGivenTopics =
      numTopics * (lgamma(numWords * beta) - numWords * lgamma(beta)) -
        denominator + likAggregator.likWordsGivenTopics

    val likTopics =
      numDocs * (lgamma(numTopics * alpha) - numTopics * lgamma(alpha)) +
        likAggregator.likTopics

    likWordsGivenTopics + likTopics
  }
}
