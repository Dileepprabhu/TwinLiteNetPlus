import cv2
import numpy as np
import os
import shutil
import time  # timing remains
import onnxruntime as ort  # added for ONNX inference
import argparse  # added for argument parsing

def Run(session, img, lane_only=False):
    img = cv2.resize(img, (640, 384))
    img_rs = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    input_name = session.get_inputs()[0].name
    if lane_only:
        outputs = session.get_outputs()
        # If more than one output exist, assume the lane branch is the second output.
        if len(outputs) > 1:
            output_names = [outputs[1].name]
        else:
            output_names = [outputs[0].name]
        img_out = session.run(output_names, {input_name: img})
        # Process lane predictions only – blue overlay for lanes.
        ll_predict = np.argmax(img_out[0], axis=1)
        LL = (ll_predict[0].astype(np.uint8)) * 255
        img_rs[LL > 100] = [0, 0, 255]
    else:
        # Original processing: two outputs for drivable area and lane predictions
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX inference with optional lane-only mode")
    parser.add_argument('--lane_only', action='store_true', help="Enable lane-only mode")
    args = parser.parse_args()

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
        img = Run(session, img, lane_only=args.lane_only)
        elapsed_time = time.time() - start_time
        print(f"Image {imgName}: Inference time = {elapsed_time:.4f} seconds", flush=True)
        cv2.imwrite(os.path.join('results', imgName), img)