"""
Image Captioning test harness for Hugging-Face pre-trained models, processors, and tokenizers
can be used as:
* command-line tool: `python caption.py`
* python module: `import caption`
"""
from io import BytesIO
from os import _exit, lstat, path
from PIL import Image, UnidentifiedImageError
from re import compile as _compile
from requests import get as requests_get
from stat import filemode
from sys import argv
from time import perf_counter
from torch import cuda, device
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BlipForConditionalGeneration, VisionEncoderDecoderModel
from urllib.parse import unquote

MAX_IDS_CNT = 50    # model-generated IDs from pixel values
url_patt = _compile(r'(?i)^https?://')  # used to match image URLs
model_names = ["git_base", "git_large", "blip_base", "blip_large",
               "vitgpt"]  # TODO: find way to DRY this since it's dup'd below


def usage():
    "output and exit when called as cmdline script on error"
    print(f"usage: {argv[0]} <model_name> <image_0> [image_1] ... [image_n]")
    print("\tmodel_name must be one of ", repr(model_names))
    _exit(0)


def generate_caption(_device: device, processor, model, image: Image.Image, tokenizer=None) -> str:
    "takes model parameters & an image, returns a caption"
    inputs = processor(images=image, return_tensors="pt").to(_device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=MAX_IDS_CNT)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption.strip()


def getImages(image_paths: list) -> list:
    "takes array of image paths, returns array of Pillow images"
    images = []
    for image_path in image_paths:
        i_image = None
        if url_patt.match(image_path):
            resp = requests_get(image_path)
            if resp.ok:
                i_image = Image.open(BytesIO(resp.content))
            else:
                print(
                    f'fetching "{image_path}" failed with status code {resp.status_code}, skipping..')
                continue
        elif path.isfile(image_path):
            try:
                i_image = Image.open(image_path)
            except PermissionError as e:
                fm = filemode(lstat(image_path).st_mode)
                print(f'"{image_path}" is a file but is not readable (filemode:{fm}), skipping..')
                continue
            except (FileNotFoundError, UnidentifiedImageError, ValueError, TypeError) as e:
                print(f'"{e}", skipping..')
                continue
        else:
            print(f'"{image_path}" does not seem to be a file or URL, skipping..')
            continue

        print(f'"{image_path}": color mode {i_image.mode} (converting to RGB as necessary)')
        i_image = i_image.convert(mode="RGB")   # be paranoid & just always convert

        images.append(i_image)
    return images


def loadModel(model_name: str) -> list:
    "loads a captioning model into memory, returns model parameters"
    model = processor = tokenizer = None

    if model_name == 'git_base':
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    elif model_name == 'git_large':
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
    elif model_name == 'blip_base':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base")
    elif model_name == 'blip_large':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large")
    elif model_name == 'vitgpt':
        processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    else:
        return None
    return model, processor, tokenizer


def main():
    "main function called when running command-line tool"
    if len(argv) < 3:  # TODO: improve cmdline option parsing / error-checking
        usage()
    model_name = argv[1]
    image_paths = [unquote(fn.strip("\"'")) for fn in argv[2:]]

    print('Captioning images: ', repr(image_paths))
    images = getImages(image_paths)

    if len(images) == 0:
        print("No usable images, nothing ToDo,.. quitting.")
        _exit(0)
    print('Loading & Importing modules, models, processors, and tokenizers...')

    model_items = loadModel(model_name)
    if model_items is None:
        usage()
    model, processor, tokenizer = list(model_items)

    # TODO: for CPU inference mode see https://huggingface.co/docs/transformers/perf_infer_cpu
    # CUDA: Nvidia or AMD ROCm GPUs
    torch_device = "cuda" if cuda.is_available() else "cpu"
    print(f'Torch device: {torch_device}')
    _device = device(torch_device)
    model.to(_device)

    captions = []
    start = perf_counter()

    for image in images:
        captions.append(generate_caption(_device, processor, model, image, tokenizer))
    total_time = perf_counter() - start

    print(f"{model_name}: completed in {total_time} seconds.")
    print(list(zip(image_paths, captions)))


if __name__ == "__main__":
    main()
