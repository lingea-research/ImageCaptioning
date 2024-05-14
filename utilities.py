"""
utility functions for image captioning
"""
from sys import stdin
from urllib.parse import quote


def urlencode_stdin():
    "reads lines from STDIN and writes them stripped URL-encoded to STDOUT"
    for l in stdin:
        print(quote(l.strip()))
