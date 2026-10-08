import numpy as np

SEED = (0xa1d31f49, 0x857194d4, 0x4a82ab01, 0xe3b21af0)
M = 0xffffffff


class Xorshift128:
    def __init__(self, seed=SEED):
        self.x, self.y, self.z, self.w = seed

    def next(self):
        t = (self.x ^ (self.x << 11)) & M
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19) ^ t ^ (t >> 8)) & M
        return self.w

    def gauss(self, mean=0.0, scale=1.0):
        s = sum(self.next() for _ in range(12))
        return ((s - 25769803770.0) * scale) / 4294967295.0 + mean


class Lanes32:
    def __init__(self):
        self.state = (np.arange(32, dtype=np.uint64) + 0x92d68ca2).astype(np.uint32)

    def block(self):
        acc = np.zeros(32, np.float32)
        s = self.state
        for _ in range(12):
            s = s ^ (s << np.uint32(13))
            s = s ^ (s >> np.uint32(17))
            s = s ^ (s << np.uint32(5))
            acc = (acc + s.view(np.int32).astype(np.float32)).astype(np.float32)
        self.state = s
        return acc / np.float32(4294967296.0)

    def draw(self, n):
        blocks = [self.block() for _ in range((n + 31) // 32)]
        return np.concatenate(blocks)[:n]
