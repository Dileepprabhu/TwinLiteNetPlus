import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model.model import TwinLiteNetPlus as net  # Updated import to match the provided file
import cv2
import time  # added for timing
import argparse  # added for argument parsing

def Run(model, img):
    img = cv2.resize(img, (640, 384))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.float() / 255.0
    # Invoke the model with only_lane=True
    with torch.no_grad():
        img_out = model(img)
        lane_out = img_out[1]  # use the second output for lane predictions
        _, ll_predict = torch.max(lane_out, 1)
    # Process only lane predictions; remove drivable area branch
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[LL > 100] = [0, 0, 255]  # Blue overlay for lane detection
    return img_rs

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run TwinLiteNet lane-only inference")
    parser.add_argument('--data_parallel', action='store_true', help="Wrap the model in DataParallel")
    args = parser.parse_args()

    # Instantiate model with lane_only=True
    model = net(lane_only=True)
    
    # Optionally wrap model in DataParallel
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    
    # Load and adjust state_dict keys based on --data_parallel flag
    state_dict = torch.load('pretrained/small.pth', map_location=torch.device('cpu'))
    if args.data_parallel:
        if not list(state_dict.keys())[0].startswith("module."):
            new_state_dict = {"module." + k: v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
    else:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Remove keys for drivable area branch if lane_only mode is enabled
    if model.lane_only:
        keys_to_remove = [k for k in new_state_dict.keys() if k.startswith("up_1_da") 
                                                     or k.startswith("up_2_da") 
                                                     or k.startswith("out_da")]
        for k in keys_to_remove:
            del new_state_dict[k]
    model.load_state_dict(new_state_dict)
    model.eval()

    image_list = os.listdir('inference/images')
    print(image_list)
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.mkdir('results')

    for i, imgName in enumerate(image_list):
        img = cv2.imread(os.path.join('inference', 'images', imgName))
        start_time = time.time()  # start timer
        img = Run(model, img)
        elapsed_time = time.time() - start_time  # end timer and compute elapsed time
        print(f"Image {imgName}: Inference time = {elapsed_time:.4f} seconds", flush=True)
        cv2.imwrite(os.path.join('results', imgName), img)