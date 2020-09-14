from flask import Flask, render_template, request
import pandas as pd
import re
from textblob import TextBlob
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from utils.utils import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # request was a POST
        app.vars['name'] = request.form['youtube_url']
        app.symbol = request.form['youtube_url']
        app.vars['action'] = request.form['cluster_number']
        return render_template('userinfo', youtube_url=app.vars['name'])


@app.route('/summary', methods=['GET', 'POST'])
def userinfo():
    APIKEY = 'AIzaSyA_2n3Fe3Ndw0pz-uz5iK4JgxVr1GEEFw4'

    video_url = request.form['youtube_url']
    cluster_number = request.form['cluster_number']
    base_url = 'https://www.googleapis.com/youtube/v3/'

    regex = r"v=(-|[a-zA-Z]|\d)+"

    matches = re.search(regex, video_url)
    video_id = matches[0][2:]

    video_stat_url = base_url + "videos?part=snippet%2CcontentDetails%2Cstatistics&id=" + video_id + "&key=" + APIKEY
    response = fetch_url(video_stat_url)

    video_title = response["items"][0]["snippet"]["title"]
    is_caption_available = response["items"][0]["contentDetails"]["caption"]
    video_statistics = response["items"][0]["statistics"]  # contains no of comments/ vies
    # snippet.description
    # captions GET https://www.googleapis.com/youtube/v3/captions?videoId=M7FIvfx5J10&part=snippet&key=[YOUR_API_KEY]
    if video_id == '':
        return render_template('noinput.html')
    else:
        comment_url = 'https://www.googleapis.com/youtube/v3/commentThreads?&part=snippet,replies&videoId=' + video_id + '&key=' + APIKEY
        comments = fetch_comments(comment_url=comment_url)

        # TODO clean text

        corpus = comments
        embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')

        corpus_embeddings = embedder.encode(corpus)
        num_clusters = int(cluster_number)
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        aclusters = show_clusters(cluster_assignment, corpus)

        res = []
        for k, v in aclusters.items():
            dataFrame = pd.DataFrame([(s, TextBlob(s).sentiment) for s in v], columns=["text", "sentiment"])
            dataFrame['score'] = dataFrame['sentiment'].apply(lambda x: abs(x[0]) + abs(x[1]))
            t = dataFrame.loc[dataFrame["score"].idxmax()]["text"]
            res.append(t)

        # render template
        html = render_template(
            'summaryRes.html',
            video_title=video_title,
            src_video="https://www.youtube.com/embed/" + video_id,
            number_of_comments=video_statistics["commentCount"],
            ticker=video_id,
            message=video_id,
            results=res
        )

        return html


@app.errorhandler(500)
def page_not_found(e):
    return render_template('ConnectionError.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
#    app.run(debug=True, use_reloader=True)
