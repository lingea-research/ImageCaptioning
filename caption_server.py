"""
web service interface to caption.py for Image Captioning
depending on memory constraints may need to accept only one connection at a time
when a model is loaded it will continue to be used until changed
"""
import logging

from flask import abort, jsonify, make_response, Flask, Response, redirect, request, send_from_directory
from flask_cors import CORS
from io import BytesIO
from os import _exit, listdir, path
from PIL import Image
from re import compile as _compile
from requests import get as requests_get
from sys import argv
from time import perf_counter
from torch import cuda, device
from urllib.parse import unquote
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from zipfile import ZipFile, ZipInfo

import caption

app = Flask(__name__, static_url_path='')

curr_model = ""
model = processor = tokenizer = None
torch_device = "cuda" if cuda.is_available() else "cpu"
_device = device(torch_device)
IMAGES_DIR = path.join(app.static_folder, 'img')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # TODO: any others?

CORS(app, resources=r'/*', supports_credentials=True, origins='*')  # TODO: make this configurable


def getImagesList() -> list:
    "returns list of images in default image directory"
    return listdir(IMAGES_DIR)


def JSON_abort(code: int, msg: str) -> Response:
    "takes an HTTP code and error message and returns a JSON response"
    return make_response(jsonify(msg=msg), code)


@app.route("/", methods=['GET'])
def defaultRoute() -> Response:
    "serves a simple web interface for testing much of the API functionality"
    return app.send_static_file('index.html')


@app.route('/api/list_models', methods=['GET'])
def listModels() -> Response:
    "returns JSON Response with available model names from 'caption.py' & current model loaded"
    global curr_model
    return jsonify({'models': caption.model_names, 'curr_model': curr_model})


@app.route('/api/list_images', methods=['GET'])
def listImages() -> Response:
    "returns JSON Response with list of images in default images directory"
    return jsonify({'images': getImagesList()})


@app.route('/api/load_model/<req_model>', methods=['GET'])
def loadModel(req_model: str) -> Response:
    "takes requested model name & returns JSON Response with available models & current loaded"
    global curr_model
    if not _loadModel(req_model):
        return JSON_abort(404, f'model "{req_model}" not found')
    return jsonify({'models': caption.model_names, 'curr_model': curr_model})


def _loadModel(req_model: str) -> bool:
    "takes requested model and tries load if not current, returns false if doesn't exist"
    global curr_model, _device, model, processor, tokenizer
    if req_model not in caption.model_names:
        return False
    if req_model != curr_model:
        curr_model = req_model
        app.logger.info(f'{__name__}: loading model "{curr_model}"')
        model, processor, tokenizer = caption.loadModel(curr_model)
        model.to(_device)
    return True


def allowed_file(filename: str) -> bool:
    "checks whether an uploaded file has an allowed extension"
    # https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload() -> Response:
    """
    takes a multi-part request and processes 1 image file
    takes optional parameter "req_model". If not curr_model (default), loads requested model
    If no model is currently loaded returns 412 Precondition Failed
    takes optional parameter "save_image". If set, returns a redirect to uploaded file
    if "save_image" not set (default) upload not saved but processed and caption returned
    """
    global curr_model
    req_model = request.args.get("req_model")
    if req_model is not None:
        if not _loadModel(req_model):
            return JSON_abort(404, f'model "{req_model}" not found')
    f = request.files["image"]
    if f is None or not allowed_file(f.filename):
        return JSON_abort(422, "upload rejected")
    if request.args.get("save_image") is not None:
        filename = secure_filename(f.filename)
        f.save(path.join(IMAGES_DIR, filename))
        return redirect(f"/?image={f.filename}")
    if curr_model == "":
        return JSON_abort(412, "no model loaded, cannot process request")
    captions = getCaptions(f)
    # TODO: more than 1 upload & more than 1 caption
    return jsonify({"curr_model": curr_model, "captions": [(f.filename, captions[0])]})


def getCaptions(caption_images: list | FileStorage) -> list:
    "takes list of Pillow images or FileStorage object, returns list of captions in order"
    global _device, model, processor, tokenizer
    pillows = caption.getImages(caption_images) if isinstance(
        caption_images, list) else [Image.open(caption_images)]
    captions = []
    for p in pillows:
        captions.append(caption.generate_caption(_device, processor, model, p, tokenizer))
    return captions


@app.route('/api/caption', methods=['GET'])
def captionImages() -> Response:
    "takes Request query params list 'image', captions them with current model & returns JSON Response with captions"
    if curr_model == "":
        return JSON_abort(422, "no model has been loaded")
    caption_images = []
    directory_images = getImagesList()
    for img in request.args.getlist('image'):
        if caption.url_patt.match(img):
            caption_images.append(img)
        elif img in directory_images:
            caption_images.append(path.join(IMAGES_DIR, img))
        else:
            return JSON_abort(404, f'image "{img}" not found')
    captions = getCaptions(caption_images)
    return jsonify(list(zip(request.args.getlist('image'), captions)))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
else:
    # https://trstringer.com/logging-flask-gunicorn-the-manageable-way/
    # TODO: make logging configurable
    gunicorn_logger = logging.getLogger('gunicorn.error')
    logging.getLogger('flask_cors').level = logging.ERROR
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info(f'{__name__}: Torch device is {torch_device}')
