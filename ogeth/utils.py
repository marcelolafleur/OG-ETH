import socket


def is_connected():
    """True if the internet is reachable."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=5):
            return True
    except OSError:
        return False
