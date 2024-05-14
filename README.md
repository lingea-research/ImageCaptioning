# Image Captioning Test

## Installation

1. Clone the repository.
2. Create the python virtual environment in the repository directory (optional)
	- python: `python3 -m venv ./venv`
3. Install the required libraries
	- pip: `venv/bin/pip3 install -r requirements.txt`
4. Run the _init.sh_ script to create the file directories.
5. Run the scripts using the _python3_ executable in the _venv/bin/_ directory.


Test generation of image captions using the _caption.py_ script. This script exercises a set of pre-trained open-source neural network models on a list of sample images. Images should be in JPEG or PNG format, preferably in RGB colorspace mode. This script will use hardware acceleration on machines having Nvidia CUDA or AMD ROCm GPUs, but can also run on any CPU in "cpu" mode. This mode is configured automatically.

Running the Installation steps in Section #1 (above) installs everything except PyTorch. If you want to support Nvidia or AMD GPUs (and possibly Mac M1 GPUs), install your platform-specific version of PyTorch. It's also possible to run in CPU-only mode. Refer to [Get Started | PyTorch](https://pytorch.org/get-started/) for further instructions, as the following seems to change:

**AMD ROCm**

`./venv/bin/pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.6`

**Nvidia CUDA**

`./venv/bin/pip3 install torch`

**CPU-only**

`./venv/bin/pip3 install torch --index-url https://download.pytorch.org/whl/cpu`

The shell script, _run_caption_tests.sh_, will test all models on all images in a given directory. Run it from the "datatoolkit" base directory:

`./run_caption_tests.sh img`

It is also possible to run _caption.py_ directly on any one model and a list of image(s):

`./venv/bin/python3 caption.py vitgpt img/climb.jpg img/jetski.png img/city.jpg`

Finally, a simple web interface can also be run:

`./run_caption_web.sh`

This will run a persistent web server on localhost port 8080 (or whichever port was set on command-line), that will keep in memory whichever model was loaded last, ready to run. It prints log messages to the terminal and can be stopped with ctrl-c. By default, images are listed from the "img/" directory. Point your browser to:

[localhost:8080](http://localhost:8080/)

to use the web interface.

### Available pre-trained models (in increasing order by total model file size)
| name       | size      | webpage                                                                                                |
-------------|-----------|--------------------------------------------------------------------------------------------------------|
| git_base   | **676MB** | [microsoft/git-base-coco](https://huggingface.co/microsoft/git-base-coco)                               |
| vitgpt     | **941MB** | [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)     |
| blip_base  | **946MB** | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)   |
| git_large  | **1.5GB** | [microsoft/git-large-coco](https://huggingface.co/microsoft/git-large-coco)                             |
| blip_large | **1.8GB** | [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) |

**Beware: upon each model first-run, model files will be downloaded to a cache on the local computer (~/.cache/huggingface/hub by default). The total size of all model files is just over 6GB**

More open-source models may be added in future versions.

