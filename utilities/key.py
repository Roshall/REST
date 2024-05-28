import hashlib


def generate_key(key: str) -> str:
    return hashlib.md5(key.encode('utf-8')).hexdigest()
