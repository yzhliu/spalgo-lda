package org.apache.spark.mllib.spalgo.lda

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, sum => brzSum}
import breeze.stats.distributions.Multinomial
import org.apache.spark.graphx.{EdgeTriplet, VertexId, Graph, Edge}
import org.apache.spark.mllib.linalg.{Vector => SV, Vectors}
import org.apache.spark.mllib.spalgo.lda.components.TopicCount
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer
import scala.util.Random

class CgsLda extends Serializable {
  private type VD = Array[Int]
  private type ED = Array[Int]
  // edge assignments
  private type Msg = TopicCount

  private var numTopics: Int = _
  private var numWords: Int = _
  private var alpha: Double = _
  private var beta: Double = _

  // vertex-data: (should be atomic) count of appearance of each topic
  // edge-data: the topic assigned to every appearance of the word in this document
  def load(documents: RDD[(Long, SV)], numTopics: Int): Graph[VD, ED] = {
    this.numTopics = numTopics
    this.numWords = documents.first()._2.size
    val edges = documents.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    val graphInit = Graph.fromEdges(edges, null, edgeStorageLevel = StorageLevel.MEMORY_AND_DISK)
    graphInit.mapVertices((vertexId, _) => Array.ofDim[Int](numTopics))
  }

  def train(documents: RDD[(Long, SV)]) = {
    val graph = load(documents, numTopics)
    val trainedGraph = graph.pregel(new TopicCount(numTopics))(vertexUpdater, sendMessage, mergeMessage)
  }

  private def vertexUpdater(vertexId: VertexId, vertexData: VD, message: Msg): VD = {
    message.topics
  }

  private def sendMessage(triplet: EdgeTriplet[VD, ED]): Iterator[(VertexId, Msg)] = {
    /*
    if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
      Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
    } else {
      Iterator.empty
    }
    */
    val wordTopicCount = triplet.srcAttr
    val docTopicCount = triplet.dstAttr
    require(docTopicCount.length == numTopics)
    require(wordTopicCount.length == numTopics)
    // run the actual gibbs sampling
    val prob = BV.zeros[Double](numTopics)
    val assignment = triplet.attr
    //edge.data().nchanges = 0
    assignment.foreach(topicId => {
      val oldTopicId = topicId
      //if (asg != NULL_TOPIC) {
        // construct the cavity
        docTopicCount(topicId) -= 1
        wordTopicCount(topicId) -= 1
        //--GLOBAL_TOPIC_COUNT[asg]
      //}
      for (t <- 0 until numTopics) {
        val n_dt = Math.max(docTopicCount(t), 0).toDouble
        val n_wt = Math.max(wordTopicCount(t), 0).toDouble
        val n_t = 0.0
        //val n_t = Math.max(globalTopicCount(t), 0).toDouble
        prob(t) = (alpha + n_dt) * (beta + n_wt) / (beta * numWords + n_t)
      }
      val rnd = Multinomial(prob)
      val asg = rnd.draw()
      // asg = std::max_element(prob.begin(), prob.end()) - prob.begin();
      docTopicCount(asg) += 1
      wordTopicCount(asg) += 1
      //++ GLOBAL_TOPIC_COUNT[asg]
      if (asg != oldTopicId) {
        //++ edge.data().nchanges;
        //INCREMENT_EVENT(TOKEN_CHANGES, 1);
      }
    }) // End of loop over each token
    // singla the other vertex
    //context.signal(get_other_vertex(edge, vertex));
    Iterator.empty
  }

  private def mergeMessage(msgA: Msg, msgB: Msg): Msg = {
    msgA.add(msgB)
  }

  private def initializeEdges(
     rnd: Random,
     words: SV,
     docId: Long,
     numTopics: Int) = {//: Iterator[Edge[ED]] = {
    require(docId >= 0, s"Invalid document id $docId")
    val vertexDocId = -(docId + 1L)
    val edges = ListBuffer[Edge[Array[Int]]]()
    words.foreachActive {
      case (wordId, counter) if counter > 0.0 =>
        val vertexWordId = wordId + 1L
        val topics = Array.ofDim[Int](counter.toInt)
        for (i <- 0 until counter.toInt) {
          topics(i) = rnd.nextInt(numTopics)
        }
        edges += Edge(vertexWordId, vertexDocId, topics)
      case _ =>
    }
    edges
  }
}
