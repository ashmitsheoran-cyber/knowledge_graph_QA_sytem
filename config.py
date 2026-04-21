import os
import ssl
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()

def apply_corporate_patches():
    # 1. SSL Unverified Context for corporate proxies
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    
    # 2. Disable Warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 3. Global Session Patch for Requests
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request

# API Keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GRAPH_FILE = "knowledge_graph.json"