from flask import Flask, jsonify, request, redirect, render_template
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        # class_id, class_name = get_prediction(image_bytes=img_bytes)
        # class_name = format_class_name(class_name)
        # return render_template('result.html', class_id=class_id,
        #                        class_name=class_name)
    return render_template('index.html')