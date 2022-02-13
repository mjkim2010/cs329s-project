from flask import Flask, jsonify, request, redirect, render_template
from scene_clustering_utils import BasicPlacesDataset, ImageClusterer
from IQA_model_utils import ImageDataset, load_models, score_image
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files:
            return
        pil = [Image.open(f) for f in files]
        dataset = BasicPlacesDataset(pil)
        images = [dataset[i] for i in range(len(files))]
        pil_standardized = [transforms.ToPILImage()(t) for t in images]
        
        model_scene = ImageClusterer()
        # manually set k
        k = 2
        clusters = model_scene(pil_standardized, k)
        print(f'clusters: {clusters}')
        print(f'file names: {files}')

        cluster_imgs = {} #k=cluster id, v=list of img indices
        for i, c_id in enumerate(clusters):
            if c_id not in cluster_imgs:
                cluster_imgs[c_id] = []
            cluster_imgs[c_id].append(i)

        dataset_iqa = ImageDataset(images)
        model_IQA, clf = load_models()
        cluster_ratings = {} #k=cluster id, v=list of ratings for cluster
        for c_id in range(k):
            if c_id not in cluster_imgs: 
                continue
            imgs = torch.stack([dataset_iqa[idx] for idx in cluster_imgs[c_id]])
            ratings = score_image(model_IQA, clf, imgs).tolist() # converting to list to make printing easier in html
            cluster_ratings[c_id] = ratings
        print(f'clusterID to ratings dict: {cluster_ratings}')
        return render_template('result.html', clusters=clusters,
                               file_names=files, cluster_ratings=cluster_ratings)
    return render_template('index.html')