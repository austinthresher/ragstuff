import os

# TODO: All of these should be overridable from the environment

DEFAULT_MODEL = 'google/gemini-2.5-flash:nitro'
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Right now this is a SearXNG instance
WEB_SEARCH_URL = "http://localhost:16000"

# Don't return websites that contain these strings because they
# won't have any actual content
BLOCKED_STRINGS = [
    "website is running Anubis",
]

