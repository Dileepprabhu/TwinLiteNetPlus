import cv2
import numpy as np
import os
import shutil
import time  # timing remains
import onnxruntime as ort  # added for ONNX inference

def Run(session, img):
    img = cv2.resize(img, (640, 384))
    img_rs = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    img_out = session.run(output_names, {input_name: img})
    x0, x1 = img_out  # assuming outputs "da" and "ll"
    da_predict = np.argmax(x0, axis=1)
    ll_predict = np.argmax(x1, axis=1)
    DA = (da_predict[0].astype(np.uint8)) * 255
    LL = (ll_predict[0].astype(np.uint8)) * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]
    return img_rs

# Create an ONNX runtime session using the exported ONNX model
# Enable GPU execution provider if available
session = ort.InferenceSession('pretrained/small.onnx', providers=['CUDAExecutionProvider'])

# ...existing code for handling result folder...
image_list = os.listdir('inference/images')
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')
for imgName in image_list:
    img = cv2.imread(os.path.join('inference/images', imgName))  # updated folder path
    start_time = time.time()
    img = Run(session, img)
    elapsed_time = time.time() - start_time
    print(f"Image {imgName}: Inference time = {elapsed_time:.4f} seconds", flush=True)
    cv2.imwrite(os.path.join('results', imgName), img)