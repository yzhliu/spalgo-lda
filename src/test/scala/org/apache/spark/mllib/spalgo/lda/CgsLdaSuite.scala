package org.apache.spark.mllib.spalgo.lda

import org.apache.spark.mllib.linalg.{Vector => SV, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest._

class CgsLdaSuite extends FunSuite with BeforeAndAfterAll with ShouldMatchers {
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[1]")
      .setAppName("CgsLdaTest")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  test("loading documents as graph") {
    val documents = sc.parallelize(Array(
      (0L, Vectors.dense(1.0, 0.0, 3.0)),
      (1L, Vectors.dense(0.0, 2.0, 1.0))))
    val numTopics = 2
    val cgsLda = new CgsLda(50.0/numTopics, 0.01, numTopics)
    val graph = cgsLda.load(documents)

    val edges = graph.edges.collect()
    val vertices = graph.vertices.collect().toMap
    assert(edges.length == 4)
    assert(vertices.size == 5)

    cgsLda.train(documents, 20)

    /*
    val messages = cgsLda.nextIteration()
    messages.collect().foreach { case (vertexId, message) =>
      val vertexType = if (vertexId < 0) "document" else "word"
      println(s"Old counts for $vertexType($vertexId): ${vertices(vertexId)}")
      println(s"New counts for $vertexType($vertexId): $message")
    }
    */
  }
}
