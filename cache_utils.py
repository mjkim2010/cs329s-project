#!/usr/bin/env python3
import hashlib
from collections import OrderedDict

class SimpleLRUCache:
    def __init__(self, maxsize):
        self.cache = OrderedDict()
        self.cache_max_size = maxsize
    def __len__(self):
        return len(self.cache)
    def __contains__(self, key):
        return key in self.cache
    def __getitem__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
        return self.cache[key]
    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cache_max_size:
            self.cache.popitem(last=False)

def get_img_hash(img_fp):
    with open(img_fp, 'rb') as f:
        img_hash = hashlib.sha256(f.read()).hexdigest()
    return img_hash
