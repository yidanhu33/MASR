from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler
from .all_in import All_in_NegativeSampler

NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
    All_in_NegativeSampler.code():All_in_NegativeSampler,
}

def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, seed, save_folder,mode):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count, item_count, sample_size, seed, save_folder,mode)
