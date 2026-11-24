import requests
import urllib3
import ssl
import socket


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    """Transport adapter with a custom SSL context, for the UN Data Portal,
    which lacks RFC 5746 renegotiation (rejected by OpenSSL 3). Use
    `get_legacy_session()` instead of `requests` directly.
    """

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
        )


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    # OP_LEGACY_SERVER_CONNECT (ssl.OP_LEGACY_SERVER_CONNECT in 3.12+)
    ctx.options |= 0x4
    session = requests.session()
    session.mount("https://", CustomHttpAdapter(ctx))
    return session


# True if the internet is reachable
def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False
