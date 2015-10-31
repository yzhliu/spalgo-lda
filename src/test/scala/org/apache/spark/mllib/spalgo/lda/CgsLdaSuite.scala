package org.apache.spark.mllib.spalgo.lda

import org.apache.spark.graphx.EdgeContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest._
import org.apache.spark.mllib.linalg.{Vector => SV, Vectors, DenseVector}

class CgsLdaSuite extends FunSuite with BeforeAndAfterAll with ShouldMatchers {
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
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
    val cgsLda = new CgsLda
    val graph = cgsLda.load(documents, 2)
    val edges = graph.edges.collect()
    println(graph.edges.collect().mkString(", "))
    println(graph.vertices.collect().mkString(", "))
    println("--------Aggregating-------------")

    val vertexRDD = cgsLda.aggregate(graph)
    println(graph.vertices.collect().mkString(", "))
  }
}
