class baseConfig(object):
    DEBUG = True
    SECRET_KEY = "get-stronger"
    CACHE_TYPE: "SimpleCache",  # Flask-Caching related configs
    CACHE_DEFAULT_TIMEOUT: 10,
