from datapizza.type import DenseEmbedding, SparseEmbedding

from datapizza.embedders.fastembedder import FastDenseEmbedder, FastEmbedder


def test_init_fastembedder():
    embedder = FastEmbedder(model_name="Qdrant/bm25")
    assert embedder is not None


def test_fastembedder_embed_single():
    embedder = FastEmbedder(model_name="Qdrant/bm25")
    result = embedder.embed("hello world")
    assert isinstance(result, SparseEmbedding)
    assert result.name == "Qdrant/bm25"
    assert isinstance(result.values, list)
    assert isinstance(result.indices, list)
    assert len(result.values) == len(result.indices)


def test_fastembedder_embed_batch():
    embedder = FastEmbedder(model_name="Qdrant/bm25")
    result = embedder.embed(["hello world", "foo bar"])
    assert isinstance(result, list)
    assert len(result) == 2
    for r in result:
        assert isinstance(r, SparseEmbedding)


def test_fastembedder_custom_embedding_name():
    embedder = FastEmbedder(model_name="Qdrant/bm25", embedding_name="custom")
    result = embedder.embed("hello world")
    assert result.name == "custom"


def test_init_dense_fastembedder():
    embedder = FastDenseEmbedder(model_name="BAAI/bge-small-en-v1.5")
    assert embedder is not None


def test_dense_fastembedder_embed_single():
    embedder = FastDenseEmbedder(model_name="BAAI/bge-small-en-v1.5")
    result = embedder.embed("hello world")
    assert isinstance(result, DenseEmbedding)
    assert result.name == "BAAI/bge-small-en-v1.5"
    assert isinstance(result.vector, list)
    assert len(result.vector) > 0


def test_dense_fastembedder_embed_batch():
    embedder = FastDenseEmbedder(model_name="BAAI/bge-small-en-v1.5")
    result = embedder.embed(["hello world", "foo bar"])
    assert isinstance(result, list)
    assert len(result) == 2
    for r in result:
        assert isinstance(r, DenseEmbedding)
        assert len(r.vector) > 0


def test_dense_fastembedder_custom_embedding_name():
    embedder = FastDenseEmbedder(
        model_name="BAAI/bge-small-en-v1.5", embedding_name="custom"
    )
    result = embedder.embed("hello world")
    assert result.name == "custom"
