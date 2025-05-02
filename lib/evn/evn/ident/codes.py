import hashlib

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def hash(obj, letters=3) -> str:
    if isinstance(obj, str): s = obj
    elif isinstance(obj, (int, float)): s = str(obj)
    else: s = str(id(obj))
    h = hashlib.sha256(s.encode()).digest()
    num = int.from_bytes(h[:2], 'big')  # or use more bytes if needed
    num = num % (26**letters)
    chars = []
    for _ in range(letters):
        chars.append(alphabet[num % 26])
        num //= 26

    return ''.join(reversed(chars))
