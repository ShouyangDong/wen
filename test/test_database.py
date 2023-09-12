from elasticsearch import Elasticsearch
import pickle

es = Elasticsearch(
    ["https://10.100.158.12:9200"],
    verify_certs=False,
    http_auth=("elastic", "*qdSlBZ7AmkaHhyf0VLN"),
)

with open("./../data/sentence_embedding.pkl", "rb") as fIn:
    cache_data = pickle.load(fIn)
    corpus_chunk = cache_data['corpus_embeddings']



def strings_ranked_by_keyword(query, text_embeddings=None, insert=False):
    if insert:
        for doc_content in text_embeddings:
            es.index(
                index="doc",
                body={"content": doc_content, "publish_date": datetime.now()},
            )

    res = es.search(index="doc", body={"query": {"match": {"content": query}}})
    formatted_top_results = set()
    for hit in res["hits"]["hits"]:
        formatted_top_results.add(hit["_source"]["content"])
    formatted_top_results = list(formatted_top_results)
    return formatted_top_results


if __name__ == "__main__":
    querys = ["mluOpGetGenerateProposalsV2WorkspaceSize"]
    for query in querys:
        formatted_top_results = strings_ranked_by_keyword(query)
        for doc in formatted_top_results:
            print(corpus_chunk[doc]["corpus"])