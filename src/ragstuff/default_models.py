from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.cross_encoders import BaseCrossEncoder


default_device = "cuda"


def _default_chat_model(**override_kwargs) -> BaseChatModel:
    raise NotImplementedError("No default chat model configured")

def _default_embeddings(**override_kwargs) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    kwargs = {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
    model_kwargs = {"device": default_device}
    model_kwargs.update(kwargs.pop("model_kwargs", {}))
    kwargs.update(override_kwargs)
    return HuggingFaceEmbeddings(model_kwargs=model_kwargs, **kwargs)


def _default_cross_encoder(**override_kwargs) -> BaseCrossEncoder:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    kwargs = {"model_name": "cross-encoder/ms-marco-MiniLM-L2-v2"}
    model_kwargs = {"device": default_device}
    model_kwargs.update(kwargs.pop("model_kwargs", {}))
    kwargs.update(override_kwargs)
    return HuggingFaceCrossEncoder(model_kwargs=model_kwargs, **kwargs)


embeddings = _default_embeddings
cross_encoder = _default_cross_encoder
chat_model = _default_chat_model
