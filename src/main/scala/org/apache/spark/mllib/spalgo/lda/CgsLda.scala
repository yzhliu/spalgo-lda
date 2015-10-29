package org.apache.spark.mllib.spalgo.lda

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV,
  Vector => BV, sum => brzSum}
import org.apache.spark.graphx.{Graph, Edge}
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer
import scala.util.Random

class CgsLda extends Serializable {
  // vertex-data: (should be atomic) count of appearance of each topic
  // edge-data: the topic assigned to every appearance of the word in this document
  def load(documents: RDD[(Long, SV)], numTopics: Int): Graph[Array[Int], Array[Int]] = {
    val numWords = documents.first()._2.size
    val edges = documents.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    val graphInit = Graph.fromEdges(edges, null, edgeStorageLevel = StorageLevel.MEMORY_AND_DISK)
    graphInit.mapVertices((vertexId, _) => Array[Int](numTopics))
  }

  private def initializeEdges(
     rnd: Random,
     words: SV,
     docId: Long,
     numTopics: Int) = {//: Iterator[Edge[ED]] = {
    require(docId >= 0, "Invalid document id ${docId}")
    val vertexDocId = -(docId + 1L)
    val edges = ListBuffer[Edge[Array[Int]]]()
    words.foreachActive {
      case (wordId, counter) if counter > 0.0 =>
        val vertexWordId = wordId + 1L
        val topics = new Array[Int](counter.toInt)
        for (i <- 0 until counter.toInt) {
          topics(i) = rnd.nextInt(numTopics)
        }
        edges += Edge(vertexWordId, vertexDocId, topics)
      case _ =>
    }
    edges
  }
}
