from src.services.retrieval_service import VectorStore


def test_vector_store_add_and_search(tmp_path):
    storage = tmp_path / "vs"
    storage.mkdir()
    vs = VectorStore(str(storage))
    docs = ["wheat farming best practices", "rice irrigation techniques"]
    vs.add_documents(docs)
    assert vs.get_document_count() >= 2
    results = vs.search("wheat")
    assert isinstance(results, list)
    assert len(results) >= 1
