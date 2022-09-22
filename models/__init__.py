from .bert import BERTModel
MODELS = {
    BERTModel.code(): BERTModel,
}


def model_factory(args,xbm_f,centroids_f,xbm_t):
    model = MODELS[args.model_code]
    return model(args,xbm_f,centroids_f,xbm_t )

