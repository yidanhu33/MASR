from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np
import random 

class All_in_NegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'all_in'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(1,self.user_count+1):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])
            samples = []
            total_item_list = [i for i in range(1, self.item_count+1)]
            samples = [i for i in total_item_list if i not in seen]
            random.seed(10)
            random_sample_list = []
            for _ in range(len(seen)):
                item = np.random.choice(self.item_count) + 1
                while item in seen:
                    item = np.random.choice(self.item_count) + 1
                random_sample_list.append(item)

            samples = samples + random_sample_list
            negative_samples[user] = samples
        return negative_samples
