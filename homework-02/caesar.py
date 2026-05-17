import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    for char in plaintext:
        if 'a' <= char <= 'z':
            ciphertext += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= char <= 'Z':
            ciphertext += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            ciphertext += char
    return ciphertext




def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    for char in ciphertext:
        if 'a' <= char <= 'z':
            plaintext += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
        elif 'A' <= char <= 'Z':
            plaintext += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
        else:
            plaintext += char
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    max_matches = 0
    
    for shift in range(26):
        decrypted = decrypt_caesar(ciphertext, shift)
        words = decrypted.lower().split()
        matches = sum(1 for word in words if word.strip(".,!?") in dictionary)
        
        if matches > max_matches:
            max_matches = matches
            best_shift = shift
            
    return best_shift
