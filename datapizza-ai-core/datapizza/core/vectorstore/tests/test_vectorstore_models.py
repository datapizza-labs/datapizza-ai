import pytest

from datapizza.core.vectorstore import VectorConfig
from datapizza.type import EmbeddingFormat


def test_vector_config_model_post_init():
    with pytest.raises(ValueError):
        VectorConfig(name="dense_emb_name", format=EmbeddingFormat.DENSE)
