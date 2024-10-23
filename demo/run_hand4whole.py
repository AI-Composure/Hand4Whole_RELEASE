import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh
import json
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from glob import glob
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as get_detectron_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt

def get_one_box(det_output):
    max_score = 0
    max_bbox = None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = score

    return max_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.dataset_path, "Please set dataset_path."
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
dataset_path = args.dataset_path

# snapshot load
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path, weights_only=True)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

detectron_cfg = get_detectron_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
detectron_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(detectron_cfg)

# prepare input image
transform = transforms.ToTensor()
save_path = osp.join(dataset_path, 'smplx_init')
os.makedirs(save_path, exist_ok=True)
img_path_list = glob(osp.join(dataset_path, 'images', '*.png')) + glob(osp.join(dataset_path, 'images', '*.jpg'))
img_path_list = sorted(img_path_list)
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
video_save = cv2.VideoWriter(osp.join(dataset_path, 'smplx_init.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
bbox = None
for image_path in tqdm(img_path_list):
    frame_idx = image_path.split('/')[-1][:-4]
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    outputs = predictor(original_img)
    instances = outputs["instances"]
    pred_classes = instances.pred_classes
    pred_boxes = instances.pred_boxes
    person_class_id = 0
    bbox = pred_boxes[pred_classes == person_class_id]
    bbox = bbox.tensor[0]
    bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item()-bbox[0].item(), bbox[3].item()-bbox[1].item()]
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    # render mesh
    vis_img = original_img[:,:,::-1].copy()
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
    frame = np.concatenate((vis_img, rendered_img),1)
    frame = cv2.putText(frame, frame_idx, (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))

    # save SMPL-X parameters
    root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
    body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
    lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
    rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
    jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
    shape = out['smplx_shape'].detach().cpu().numpy()[0]
    expr = out['smplx_expr'].detach().cpu().numpy()[0] 
    with open(osp.join(save_path, frame_idx + '.json'), 'w') as f:
        json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                'body_pose': body_pose.reshape(-1,3).tolist(), \
                'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                'leye_pose': [0,0,0],\
                'reye_pose': [0,0,0],\
                'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                'shape': shape.reshape(-1).tolist(), \
                'expr': expr.reshape(-1).tolist()}, f)

