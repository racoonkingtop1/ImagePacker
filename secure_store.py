"""Encrypts secrets at rest using Windows DPAPI (CryptProtectData).

DPAPI ties the encrypted blob to the current Windows user account and
machine: the ciphertext stored on disk is useless without being decrypted
by the same Windows login that created it, and Windows itself manages the
underlying key material (nothing app-specific to lose or leak).
"""
import base64
import ctypes
import ctypes.wintypes as wintypes

# App-specific entropy mixed into every encrypt/decrypt call so the blob
# cannot be decrypted by unrelated tools that also call DPAPI as this user.
_ENTROPY = b"NanoBananaSender:v1:gemini-api-key"


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _to_blob(data: bytes) -> _DATA_BLOB:
    buf = ctypes.create_string_buffer(data, len(data))
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def is_available() -> bool:
    return hasattr(ctypes, "windll")


def encrypt(plaintext: str) -> str:
    """Returns a base64 string safe to store in a JSON config file."""
    if not plaintext:
        return ""
    data_in = _to_blob(plaintext.encode("utf-8"))
    entropy = _to_blob(_ENTROPY)
    data_out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(data_in), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(data_out)
    )
    if not ok:
        raise ctypes.WinError()  # get_last_error() would report 0 here: windll isn't use_last_error
    try:
        raw = ctypes.string_at(data_out.pbData, data_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(data_out.pbData)
    return base64.b64encode(raw).decode("ascii")


def decrypt(ciphertext_b64: str) -> str:
    """Returns the plaintext, or "" if the blob is empty/unreadable (e.g. it
    was created on a different machine or Windows account)."""
    if not ciphertext_b64:
        return ""
    try:
        raw = base64.b64decode(ciphertext_b64)
    except (ValueError, TypeError):
        return ""
    data_in = _to_blob(raw)
    entropy = _to_blob(_ENTROPY)
    data_out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(data_in), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(data_out)
    )
    if not ok:
        return ""
    try:
        return ctypes.string_at(data_out.pbData, data_out.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(data_out.pbData)
