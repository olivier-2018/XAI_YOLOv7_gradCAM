import os
import random
import time
import argparse
import numpy as np
from models.gradcam import YOLOV7GradCAM, YOLOV7GradCAMPP
from models.yolov7_object_detector import YOLOV7TorchObjectDetector
import cv2

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="weights/yolov7.pt", help='Path to the model')
parser.add_argument('--confidence', type=float, default=0.45, help='confidence level for object detection')
parser.add_argument('--iou-thresh', type=float, default=0.45, help='IOU threshold for object detection')

parser.add_argument('--img-path', type=str, default='figure/cam', help='input image path')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--output-dir', type=str, default='outputs/', help='output dir')
parser.add_argument('--no_text_box', action='store_true', help='do not show label and box on the heatmap')

parser.add_argument('--target-layers', type=str, default='104_act',
                    help='The layer hierarchical address to which gradcam will be applied.'
                         'Default is last layer ("104_act" for yolov7). Nb: the layer ID and name should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'gradcampp'], help='gradcam method. Default is gradcam')

parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None, 
                    help='Names of classes (list of strings). Defaults is None (coco dataset classes read directly from model). '
                        'For custom models, names should be a list of strings')

args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    
    return res_img, n_heatmat


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
        outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
        c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
        c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
    return img


# 
def main(img_path):
    """single image detection

    Args:
        img_path (string): full path to image 
    """
    
    device = args.device
    input_size = (args.img_size, args.img_size)
    
    # read image
    img = cv2.imread(img_path)  # Read image format: BGR
    print('[INFO] Loading the model')
    
    # Instantiate the YOLOv7 model and get the detection result
    model = YOLOV7TorchObjectDetector(model_weight= args.model_path, 
                                      device= device, 
                                      img_size=input_size, 
                                      names=args.names,
                                      mode='eval',
                                      confidence=args.confidence,
                                      iou_thresh=args.iou_thresh,
                                      agnostic_nms=False)
    
    # Image pre-processing Ex: Bus (1080, 810, 3) --> torch.Size([1, 3, 640, 480]) of floats values [0,1]
    torch_img = model.preprocessing(img[..., ::-1])    
    
    # Allocate a color for each class in the model
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.names]
    
    # Looping through selected model layers     
    tic = time.time()  
    for target_layer in args.target_layers.split(','):
        
        # remove any space in target_layer string
        target_layer = target_layer.replace(" ", "")
        print ("===target_layer: {} with method: {}".format(target_layer, args.method))
        
        # Get the grad-cam method
        if args.method == 'gradcam':
            saliency_method = YOLOV7GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV7GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
            
        # get prediction results
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        # result = result[..., ::-1]  # convert to bgr
        
        # Save Settings
        image_name = os.path.basename(img_path) 
        save_path = f'{args.output_dir}{image_name[:-4]}/{args.method}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'[INFO] Saving the final image at {save_path}')
        
        # Iterate through each detected class in each image
        for i, mask in enumerate(masks):
            
            # Iterate through each detected class in the image
            res_img = result.copy()
            
            # Get the location and category information of the detected class
            bbox, cls_name = boxes[0][i], class_names[0][i]
            label = f'{cls_name} {conf[0][i]}'  # category + confidence score
            
            # Get the heat map of the detected class
            res_img, heat_map = get_res_img(bbox, mask, res_img)
            res_img = plot_one_box(bbox, 
                                   res_img, 
                                   label=label, 
                                   color=colors[int(model.names.index(cls_name))],
                                   line_thickness=3)
            
            # Resize to original image size
            res_img = cv2.resize(res_img, dsize=(img.shape[:-1][::-1]))
            output_path = f'{save_path}/{target_layer[:-4]}_{i}.jpg'
            cv2.imwrite(output_path, res_img)
            print(f'{image_name[:-4]}_{target_layer[:-4]}_{i}.jpg done!!')
            
    print(f'Total time : {round(time.time() - tic, 4)} s')


if __name__ == '__main__':
    
    # case: image path is a folder
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print("List of images detected in folder: ", img_list)
        for item in img_list:
            # Get the names of the pictures in the folder in turn, and combine them into the path of the picture7
            print("===== image: {} =====".format(item))
            main(os.path.join(args.img_path, item))
    # case: single picture
    else:
        main(args.img_path)
