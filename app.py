import os
from flask import Flask, request, redirect, render_template, url_for
from scene_clustering_utils import ImageClusterer
import IQA_model_utils as IQA
import numpy as np
from torchvision import transforms
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg']) # should we add png?


def return_app():
  return app

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_scene = None # ImageClusterer()
model_IQA = None 
best_pics = {} #k=cluster id, v=idx of best quality image; for front-end use
filepaths = [] # used for rendering images in html
cluster_imgs = {} #k=cluster id, v=list of img idx
cluster_ratings = {} #k=cluster id, v=list of ratings for cluster

def regroup_imgs():
    global model_scene
    global model_IQA
    # run clustering
    clustering_method = request.form['clusteringMethod']
    if clustering_method == 'kmeans':
        if request.form['n_clusters']:
            n_clusters = int(request.form['n_clusters'])
        else:
            n_clusters = 2 # default k
        cluster(clustering_method, n_clusters=n_clusters)
    else:
        cluster(clustering_method)
    # run IQA
    run_IQA()

@app.route('/displayCluster/<clusterId>')
def show_cluster(clusterId):
    clusterId = int(clusterId)
    imgs_in_cluster = cluster_imgs[clusterId]

    imgs_with_ratings = list(zip(imgs_in_cluster, cluster_ratings[clusterId]))
    imgs_with_ratings.sort(key=lambda x: x[1], reverse=True) # sort images by decreasing quality
    # pics = ['/'+filepaths[idx] for idx, _ in imgs_with_ratings]
    pics = []
    i = 0
    while i < len(imgs_in_cluster): 
        group = []
        for j in range(2): # number of pics in row
            if i+j == len(imgs_in_cluster):
                break
            group.append(imgs_with_ratings[i+j][0]) # idx of file
        if group:
            pics.append(['/'+ filepaths[idx] for idx in group])
        i += 3
    # pics.append(['/'+ f for f in imgs_in_cluster])
    return render_template('singleCluster.html', pics=pics)

# use this route to test UI
@app.route('/bestPics', methods=['GET', 'POST'])
def display_bests():
    if request.method == 'POST':
        regroup_imgs()
        redirect(url_for('display_bests'))
    return render_template('bestpics.html', best_pics=best_pics, filepaths=filepaths)

def cluster(clustering_method, n_clusters=None):
    global model_scene
    if not model_scene: # reclustering
        model_scene = ImageClusterer()
    if clustering_method == 'kmeans':
        if not n_clusters:
            k = 2 # default k
        else:
            k = n_clusters
        clusters = model_scene.cluster_kmeans(filepaths, n_clusters=k)
    elif clustering_method == 'dbscan': # DBScan
        clusters = model_scene.cluster_dbscan(filepaths)
    else: # inside-outside pretrained
        clusters = model_scene.cluster_kmeans(filepaths, pretrained_kmeans='inside_outside')
    
    cluster_imgs.clear() 
    for i, c_id in enumerate(clusters):
        if c_id == -1: # store non-clustered items to show user (DBScan only)
            c_id = 10000
        if c_id not in cluster_imgs:
            cluster_imgs[c_id] = []
        cluster_imgs[c_id].append(i)

def run_IQA():
    global model_IQA
    if not model_IQA:
        model_IQA = IQA.IQAClass("IQAmodel")
    ratings = model_IQA(filepaths)

    cluster_ratings.clear()
    best_pics.clear()
    for c_id in cluster_imgs.keys():
        ratings_for_cluster = [ratings[i] for i in cluster_imgs[c_id]]
        cluster_ratings[c_id] = ratings_for_cluster
        best_pic_idx = cluster_imgs[c_id][np.argmax(ratings_for_cluster)]
        best_pics[c_id] = best_pic_idx

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     return redirect(request.url)
        # files = request.files.getlist('file')

        # reset global variables
        filepaths.clear()
        global model_scene
        global model_IQA

        # save uploaded images...necessary to render later
        # for f in files:
        #     if f and allowed_file(f.filename):
        #         filename = secure_filename(f.filename)
        #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #         f.save(filepath)
        #         filepaths.append(filepath)
        with os.scandir(app.config['UPLOAD_FOLDER']) as entries:
            for entry in entries:
                filename = entry.name
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                filepaths.append(filepath)
        
        # run clustering
        clustering_method = request.form['clusteringMethod']
        if clustering_method == 'kmeans':
            if request.form['n_clusters']:
                n_clusters = int(request.form['n_clusters'])
            else:
                n_clusters = 2 # default k
            cluster(clustering_method, n_clusters=n_clusters)
        else:
            cluster(clustering_method)

        # run IQA
        run_IQA()

        return redirect(url_for('display_bests'))
 
    return render_template('index.html')