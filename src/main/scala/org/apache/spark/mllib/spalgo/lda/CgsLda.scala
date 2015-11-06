package org.apache.spark.mllib.spalgo.lda

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BreezeDenseVector, Matrix => BM, SparseVector => BreezeSparseVector, Vector => BreezeVector}
import breeze.stats.distributions.Multinomial
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer
import scala.util.Random

class EdgeData(dimension: Int, numTopics: Int, rnd: Random) extends Serializable {
  val topicAssignments = Array.ofDim[Int](dimension)
  for (i <- 0 until dimension) {
    topicAssignments(i) = rnd.nextInt(numTopics)
  }

  def size = topicAssignments.length

  def apply(idx: Int) = topicAssignments(idx)

  override def toString = {
    s"TopicAssignments[${topicAssignments.mkString(",")}]"
  }
}

class VertexData(val numTopics: Int) extends Serializable {
  var topicAttachedCounts = BreezeVector.zeros[Int](numTopics)
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
    val newData = new VertexData(1)
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
  private var globalTopicCount: VertexData = _
  private var graph: Graph[VD, ED] = _

  // vertex-data: (should be atomic) count of appearance of each topic
  // edge-data: the topic assigned to every appearance of the word in this document
  def load(documents: RDD[(Long, Vector)]): Graph[VD, ED] = {
    this.numWords = documents.first()._2.size
    val edges = documents.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    graph = Graph.fromEdges(edges, null, edgeStorageLevel = StorageLevel.MEMORY_AND_DISK)
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
      = graph.aggregateMessages(sendMessage, mergeMessage).persist(StorageLevel.MEMORY_AND_DISK)
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
        wordTopicDistribution(wordId.toInt-1, t) = phi
      }
    }
    wordTopicDistribution
  }

  private def nextIteration(): Unit = {
    val prevG = graph
    // init globalTopicCount
    val messages: VertexRDD[Msg]
      = prevG.aggregateMessages(sendMessage, mergeMessage).persist(StorageLevel.MEMORY_AND_DISK)
    var newG = prevG.joinVertices(messages) {
      case (vertexId, null, newData) => newData
    }
    // calculate topic counts for words and documents
    globalTopicCount = calculateGlobalTopicCount(newG)
    newG = newG.mapTriplets((pid: PartitionID, iter: Iterator[EdgeTriplet[VD, ED]]) => {
      // sample the topic assignments z_{di}
      iter.map(triplet => {
        val wordTopicCount = triplet.srcAttr
        val docTopicCount = triplet.dstAttr
        //require(docTopicCount.length == numTopics)
        //require(wordTopicCount.length == numTopics)
        // run the actual gibbs sampling
        val prob = BreezeVector.zeros[Double](numTopics)
        val assignment = triplet.attr
        val betaSum = beta * numWords
        for (i <- 0 until assignment.topicAssignments.length) {
          val oldTopicId = assignment.topicAssignments(i)
          docTopicCount.increment(oldTopicId, -1)
          wordTopicCount.increment(oldTopicId, -1)
          globalTopicCount.increment(oldTopicId, -1)
          for (t <- 0 until numTopics) {
            prob(t) = (alpha + docTopicCount(t)) * (beta + wordTopicCount(t)) / (betaSum + globalTopicCount(t))
          }
          val rnd = Multinomial(prob)
          val newTopicId = rnd.draw()
          assignment.topicAssignments(i) = newTopicId
          docTopicCount.increment(newTopicId)
          wordTopicCount.increment(newTopicId)
          globalTopicCount.increment(newTopicId)
        } // End of loop over each token
        assignment
      })
    }, TripletFields.All)
    graph = GraphImpl(newG.vertices.mapValues(t => null), newG.edges)
    // FXIME: cause java.lang.UnsupportedOperationException:
    // Cannot change storage level of an RDD after it was already assigned a level
    //graph.persist(StorageLevel.MEMORY_AND_DISK)

    messages.unpersist(blocking = false)
    prevG.edges.unpersist(blocking = false)
    newG.unpersistVertices(blocking = false)
  }

  private def sendMessage(triplet: EdgeContext[VD, ED, Msg]): Unit = {
    val edgeData = triplet.attr
    val message = new Msg(numTopics)
    for (w <- 0 until edgeData.size) {
      message.increment(edgeData(w))
    }
    triplet.sendToSrc(message)
    triplet.sendToDst(message)
  }

  private def mergeMessage(msgA: Msg, msgB: Msg): Msg = {
    msgA + msgB
  }

  private def calculateGlobalTopicCount(g: Graph[VD, ED]): VD = {
    // words' vertexId > 0
    g.vertices.filter(t => t._1 > 0).map(_._2).aggregate(new VD(numTopics))(_ + _, _ + _)
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
}
