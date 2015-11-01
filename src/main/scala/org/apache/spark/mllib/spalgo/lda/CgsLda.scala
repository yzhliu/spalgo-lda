package org.apache.spark.mllib.spalgo.lda

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BreezeDenseVector, Matrix => BM, SparseVector => BreezeSparseVector, Vector => BreezeVector}
import breeze.stats.distributions.Multinomial
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.spalgo.lda.components.TopicCount
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

class VertexData(numTopics: Int) extends Serializable {
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
  override def toString = {
    // TODO
    s"TopicCounts[${topicAttachedCounts.toArray.mkString(",")}]"
  }
}

class CgsLda(val alpha: Double, val beta: Double) extends Serializable {
  private type VD = VertexData
  private type ED = EdgeData
  // edge assignments
  private type Msg = VertexData

  private var numTopics: Int = _
  private var numWords: Int = _
  private var globalTopicCount: VertexData = _
  private var graph: Graph[VD, ED] = _

  // vertex-data: (should be atomic) count of appearance of each topic
  // edge-data: the topic assigned to every appearance of the word in this document
  def load(documents: RDD[(Long, Vector)], numTopics: Int): Graph[VD, ED] = {
    this.numTopics = numTopics
    this.numWords = documents.first()._2.size
    val edges = documents.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    val graphInit = Graph.fromEdges(edges, null, edgeStorageLevel = StorageLevel.MEMORY_AND_DISK)
    graph = graphInit.mapVertices((vertexId, _) => new VertexData(numTopics))
    // calculate topic counts for words and documents
    val initVertexData = graph.aggregateMessages[Msg](
      triplet => {
        val edgeData = triplet.attr
        val message = new Msg(numTopics)
        for (w <- 0 until edgeData.size) {
          message.increment(edgeData(w))
        }
        println(s"Debug src=${triplet.srcId}, dst=${triplet.dstId}, edgeData=$edgeData, message=$message")
        triplet.sendToSrc(message)
        triplet.sendToDst(message)
      },
      mergeMessage)
    println("initVertexData: " + initVertexData.collect().mkString(","))
    graph = graph.joinVertices(initVertexData){ case (vertexId, oldData, newData) => newData }
    globalTopicCount = graph.vertices.filter(t => t._1 > 0).map(_._2).
      aggregate(new VD(numTopics))((a, b) => {
      a.add(b)
    }, (a, b) => { a.add(b) })
    // init globalTopicCount
    graph
  }

  def train(documents: RDD[(Long, Vector)]) = {
    val graph = load(documents, numTopics)
    //val trainedGraph = graph.pregel(new TopicCount(numTopics))(vertexUpdater, sendMessage, mergeMessage)
  }

  def nextIteration() = {
    val messages: VertexRDD[Msg] = graph.aggregateMessages(sendMessage, mergeMessage)
    messages
  }

  private def sendMessage(triplet: EdgeContext[VD, ED, Msg]): Unit = {
    val wordTopicCount = triplet.srcAttr
    val docTopicCount = triplet.dstAttr
    //require(docTopicCount.length == numTopics)
    //require(wordTopicCount.length == numTopics)
    // run the actual gibbs sampling
    val prob = BreezeVector.zeros[Double](numTopics)
    val assignment = triplet.attr
    val betaSum = beta * numWords
    assignment.topicAssignments.foreach(topicId => {
      docTopicCount.increment(topicId, -1)
      wordTopicCount.increment(topicId, -1)
      globalTopicCount.increment(topicId, -1)
      for (t <- 0 until numTopics) {
        prob(t) = (alpha + docTopicCount(t)) * (beta + wordTopicCount(t)) / (betaSum + globalTopicCount(t))
      }
      val rnd = Multinomial(prob)
      val asg = rnd.draw()
      docTopicCount.increment(asg)
      wordTopicCount.increment(asg)
      globalTopicCount.increment(asg)
    }) // End of loop over each token
    // signal the other vertex
    triplet.sendToSrc(wordTopicCount)
    triplet.sendToDst(docTopicCount)
  }

  private def mergeMessage(msgA: Msg, msgB: Msg): Msg = {
    msgA.add(msgB)
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
