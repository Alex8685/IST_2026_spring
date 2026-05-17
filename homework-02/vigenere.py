def encrypt_vigenere(plaintext, keyword):
    """
    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    keyword = keyword.upper()
    for i, char in enumerate(plaintext):
        if 'a' <= char <= 'z':
            shift = ord(keyword[i % len(keyword)]) - ord('A')
            ciphertext += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= char <= 'Z':
            shift = ord(keyword[i % len(keyword)]) - ord('A')
            ciphertext += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            ciphertext += char
    return ciphertext

def decrypt_vigenere(ciphertext, keyword):
    """
    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    keyword = keyword.upper()
    for i, char in enumerate(ciphertext):
        if 'a' <= char <= 'z':
            shift = ord(keyword[i % len(keyword)]) - ord('A')
            plaintext += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
        elif 'A' <= char <= 'Z':
            shift = ord(keyword[i % len(keyword)]) - ord('A')
            plaintext += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
        else:
            plaintext += char
    return plaintext