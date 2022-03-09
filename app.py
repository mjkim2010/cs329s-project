import os
from urllib import response
from flask import Flask, jsonify, request, redirect, render_template, url_for
from scene_clustering_utils import ImageClusterer
import IQA_model_utils as IQA
from PIL import Image
import numpy as np
from torchvision import transforms
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg']) # should we add png?

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_scene = None # ImageClusterer()
best_pics = {} #k=cluster id, v=idx of best quality image; for front-end use
filepaths = [] # used for rendering images in html
cluster_imgs = {} #k=cluster id, v=list of img idx
cluster_ratings = {} #k=cluster id, v=list of ratings for cluster

@app.route('/displayCluster/<clusterId>')
def show_cluster(clusterId):
    clusterId = int(clusterId)
    imgs_in_cluster = cluster_imgs[clusterId]

    imgs_with_ratings = list(zip(imgs_in_cluster, cluster_ratings[clusterId]))
    imgs_with_ratings.sort(key=lambda x: x[1], reverse=True) # sort images by decreasing quality
    pics = ['/'+filepaths[idx] for idx, _ in imgs_with_ratings]
    return render_template('singleCluster.html', pics=pics)

# use this route to test UI
@app.route('/bestPics')
def display_bests():
    return render_template('bestpics.html', best_pics=best_pics, filepaths=filepaths)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')

        # reset global variables
        best_pics.clear()
        filepaths.clear()
        cluster_imgs.clear()
        cluster_ratings.clear()

        # save uploaded images...necessary to render later
        for f in files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                filepaths.append(filepath)
                
        model_scene = ImageClusterer()
        clustering_method = request.form['clusteringMethod']
        if clustering_method == 'kmeans':
            if not request.form['n_clusters']:
                k = 2 # default k value if user doesn't specify
            else:
                k = int(request.form['n_clusters'])
            clusters = model_scene.cluster_kmeans(filepaths, n_clusters=k)
        elif clustering_method == 'dbscan': # DBScan
            clusters = model_scene.cluster_dbscan(filepaths)
        else: # inside-outside pretrained
            clusters = model_scene.cluster_kmeans(filepaths, pretrained_kmeans='inside_outside')

        for i, c_id in enumerate(clusters):
            if c_id == -1: # store non-clustered items to show user (DBScan only)
                c_id = 10000
            if c_id not in cluster_imgs:
                cluster_imgs[c_id] = []
            cluster_imgs[c_id].append(i)

        model_IQA = IQA.IQAClass("IQAModel")
        ratings = model_IQA(filepaths)
        
        for c_id in cluster_imgs.keys():
            ratings_for_cluster = [ratings[i] for i in cluster_imgs[c_id]]
            cluster_ratings[c_id] = ratings_for_cluster
            best_pic_idx = cluster_imgs[c_id][np.argmax(ratings_for_cluster)]
            best_pics[c_id] = best_pic_idx

        return redirect(url_for('display_bests'))
 
    return render_template('index.html')