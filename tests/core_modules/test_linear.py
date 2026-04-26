from core_modules.linear import Embedding


def test_embedding_construction():
    Embedding(10, 10, padding_idx=0)
