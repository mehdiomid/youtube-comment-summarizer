import requests
import pandas as pd
import numpy as np

def fetch_url(url):
    res = requests.get(url)
    return res.json()


def fetch_comments(comment_url):
    d = fetch_url(comment_url)
    try:
        nextPageToken = d["nextPageToken"]
    except:
        nextPageToken = None

    comments = [d["items"][index]['snippet']['topLevelComment']['snippet']['textDisplay'] for index in
                range(len(d["items"]))]

    while nextPageToken:
        comment_url = comment_url + '&pageToken=' + nextPageToken

        d = fetch_url(comment_url)
        try:
            nextPageToken = d["nextPageToken"]
        except:
            nextPageToken = None
        page_comments = [d["items"][index]['snippet']['topLevelComment']['snippet']['textDisplay'] for index in
                         range(len(d["items"]))]
        comments.extend(page_comments)

    return comments


def show_clusters(assign, corpus):
    clusters_dictionary = {}
    for i, c in enumerate(assign):
        if c in clusters_dictionary:
            clusters_dictionary[c].append(corpus[i])
        else:
            clusters_dictionary[c] = [corpus[i]]

    return clusters_dictionary

