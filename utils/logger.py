# utils/logger.py
from __future__ import annotations

import os, json, time


class JSONLLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a', buffering=1)
    def log(self, **kv):
        kv.setdefault('ts', time.time())
        self.f.write(json.dumps(kv) + '\n')
    def close(self):
        try: self.f.close()
        except: pass

