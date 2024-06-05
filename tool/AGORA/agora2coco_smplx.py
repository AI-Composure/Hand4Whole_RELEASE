import json
import torch
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import cv2
import pickle
import os
import math
import smplx
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import argparse

# projection code are modified from https://github.com/pixelite1201/agora_evaluation/blob/master/agora_evaluation/projection.py

def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel

def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1, -1, 1])
    return points

def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1, -1, -1])
    return j3d

def project_point(joint, RT, KKK):
    P = np.dot(KKK, RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]
    return joints_2d

def project_2d(
        df,
        i,
        pNum,
        joints3d,
        meanPose=False):

    dslr_sens_width = 36
    dslr_sens_height = 20.25
    imgWidth = 3840
    imgHeight = 2160 
    imgName = df['imgPath'][i]

    imgPath = df['imgPath'][i]
    if 'hdri' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0

    elif 'cam00' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 28
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene3d = True
        focalLength = 28
        camPosWorld = [
            df['camX'][i],
            df['camY'][i],
            df['camZ'][i]]
        camYaw = df['camYaw'][i]
        camPitch = 0
    
    if meanPose:
        yawSMPL = 0
        trans3d = [0, 0, 0]
    else:
        yawSMPL = df['Yaw'][i][pNum]
        trans3d = [df['X'][i][pNum],
                   df['Y'][i][pNum],
                   df['Z'][i][pNum]]

    gt2d, gt3d_camCoord, focal, princpt = project2d(joints3d, focalLength=focalLength, scene3d=scene3d,
                                    trans3d=trans3d,
                                    dslr_sens_width=dslr_sens_width,
                                    dslr_sens_height=dslr_sens_height,
                                    camPosWorld=camPosWorld,
                                    cy=imgHeight / 2,
                                    cx=imgWidth / 2,
                                    yawSMPL=yawSMPL,
                                    ground_plane=ground_plane,
                                    ind=i,
                                    pNum=pNum,
                                    meanPose=meanPose, camPitch=camPitch, camYaw=camYaw)
    return gt2d, gt3d_camCoord, focal, princpt


def project2d(
        j3d,
        focalLength,
        scene3d,
        trans3d,
        dslr_sens_width,
        dslr_sens_height,
        camPosWorld,
        cy,
        cx,
        yawSMPL,
        ground_plane,
        ind=-1,
        pNum=-1,
        meanPose=False,
        camPitch=0,
        camYaw=0):

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    camMat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    if scene3d:
        camPosWorld = unreal2cv2(
            np.reshape(
                camPosWorld, (1, 3))) + np.array(ground_plane)
    else:
        camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3)))

    # get points in camera coordinate system
    j3d = smpl2opencv(j3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]], dtype=float))

    j3d = np.matmul(rotMat, j3d.T).T
    j3d = j3d + trans3d

    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

    j3d_new = np.matmul(camera_rotationMatrix, j3d.T - camPosWorld.T).T
    j3d_new = np.matmul(camera_rotationMatrix2, j3d_new.T).T

    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    j2d = np.zeros((j3d_new.shape[0], 2))
    for i in range(j3d_new.shape[0]):
        j2d[i, :] = project_point(np.concatenate(
            [j3d_new[i, :], np.array([1])]), RT, camMat)
    
    focal = (focalLength_x, focalLength_y)
    princpt = (cx, cy)
    return j2d, j3d_new, focal, princpt

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    args = parser.parse_args()
    assert args.dataset_path, "Please set dataset_path"
    return args

args = parse_args()
dataset_path = args.dataset_path

image_id = 0
ann_id = 0
smplx_params_path = './smplx_params_cam'
cam_params_path = './cam_params'
smplx_layer = {k: smplx.create('/home/cv8/mks0601/workspace/human_model_files', 'smplx', gender=k, use_pca=False, flat_hand_mean=False) for k in ['male', 'female', 'neutral']}

smplx_joints_name= \
    ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
    'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
    'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # fingers
    'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # fingers
    'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
    'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
    'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
    'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', # finger tips
    *['Face_' + str(i) for i in range(5,56)] # face
    )
smplx_joint_part = {
            'body': list(range(smplx_joints_name.index('Pelvis'), smplx_joints_name.index('R_Eye_SMPLH')+1)) + list(range(smplx_joints_name.index('Nose'), smplx_joints_name.index('R_Heel')+1)),
            'lhand': list(range(smplx_joints_name.index('L_Index_1'), smplx_joints_name.index('L_Thumb_3')+1)) + list(range(smplx_joints_name.index('L_Thumb_4'), smplx_joints_name.index('L_Pinky_4')+1)),
            'rhand': list(range(smplx_joints_name.index('R_Index_1'), smplx_joints_name.index('R_Thumb_3')+1)) + list(range(smplx_joints_name.index('R_Thumb_4'), smplx_joints_name.index('R_Pinky_4')+1)),
            'face': list(range(smplx_joints_name.index('Face_5'), smplx_joints_name.index('Face_55')+1))}

os.makedirs(osp.join(dataset_path, smplx_params_path), exist_ok=True)
os.makedirs(osp.join(dataset_path, cam_params_path), exist_ok=True)

for split in ('train', 'validation'):
    images = []
    annotations = []
    if split == 'train':
        data_path_list = glob(osp.join(dataset_path, 'SMPLX', '*.pkl'))
    else:
        data_path_list = glob(osp.join(dataset_path, 'validation_SMPLX', 'SMPLX', '*.pkl'))
    data_path_list = sorted(data_path_list)

    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            data = {k: list(v) for k,v in data.items()}

        if split == 'train':
            img_folder_name = data_path.split('/')[-1].split('_withjv')[0] # e.g., train_0
        else:
            img_folder_name = 'validation'
        img_num = len(data['imgPath'])
        
        for i in range(img_num):
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['file_name_3840x2160'] = osp.join('images_3840x2160', img_folder_name, data['imgPath'][i])
            img_dict['file_name_1280x720'] = osp.join('images_1280x720', img_folder_name, data['imgPath'][i][:-4] + '_1280x720.png')
            images.append(img_dict)

            person_num = len(data['gt_path_smplx'][i])
            for j in range(person_num):
                ann_dict = {}
                ann_dict['id'] = ann_id
                ann_dict['image_id'] = image_id
                ann_dict['person_id'] = j
                ann_dict['smplx_param_path'] = osp.join(smplx_params_path, img_folder_name, data['imgPath'][i][:-4], str(j) + '.json')
                ann_dict['cam_param_path'] = osp.join(cam_params_path, img_folder_name, data['imgPath'][i][:-4] + '.json')
                ann_dict['gender'] = data['gender'][i][j]
                ann_dict['kid'] = data['kid'][i][j]
                ann_dict['occlusion'] = data['occlusion'][i][j]
                ann_dict['is_valid'] = data['isValid'][i][j]
                ann_dict['age'] = data['age'][i][j]
                ann_dict['ethnicity'] = data['ethnicity'][i][j]

                # bbox
                joints_2d = np.array(data['gt_joints_2d'][i][j]).reshape(-1,2)
                bbox = get_bbox(joints_2d, np.ones_like(joints_2d[:,0])).reshape(4)
                ann_dict['bbox'] = bbox.tolist()
                
                joints_2d_lhand = joints_2d[smplx_joint_part['lhand'],:]
                lhand_bbox = get_bbox(joints_2d_lhand, np.ones_like(joints_2d_lhand[:,0])).reshape(4)
                ann_dict['lhand_bbox'] = lhand_bbox.tolist()

                joints_2d_rhand = joints_2d[smplx_joint_part['rhand'],:]
                rhand_bbox = get_bbox(joints_2d_rhand, np.ones_like(joints_2d_rhand[:,0])).reshape(4)
                ann_dict['rhand_bbox'] = rhand_bbox.tolist()

                joints_2d_face = joints_2d[smplx_joint_part['face'],:]
                face_bbox = get_bbox(joints_2d_face, np.ones_like(joints_2d_face[:,0])).reshape(4)
                ann_dict['face_bbox'] = face_bbox.tolist()
                annotations.append(ann_dict)
 
                # get world, camera, and image coordinates
                with open(osp.join(dataset_path, data['gt_path_smplx'][i][j][:-4] + '.pkl'), 'rb') as f:
                    smplx_param = pickle.load(f, encoding='latin1')
                global_orient = torch.FloatTensor(smplx_param['global_orient']).view(1,-1)
                body_pose = torch.FloatTensor(smplx_param['body_pose']).view(1,-1)
                jaw_pose = torch.FloatTensor(smplx_param['jaw_pose']).view(1,-1)
                leye_pose = torch.FloatTensor(smplx_param['leye_pose']).view(1,-1)
                reye_pose = torch.FloatTensor(smplx_param['reye_pose']).view(1,-1)
                left_hand_pose = torch.FloatTensor(smplx_param['left_hand_pose']).view(1,-1)
                right_hand_pose = torch.FloatTensor(smplx_param['right_hand_pose']).view(1,-1)
                betas = torch.FloatTensor(smplx_param['betas']).view(1,-1)[:,:10]
                expression = torch.FloatTensor(smplx_param['expression']).view(1,-1)
                transl = torch.FloatTensor(smplx_param['transl']).view(1,-1)
                gender = data['gender'][i][j]
                with torch.no_grad():
                    output = smplx_layer[gender](global_orient=global_orient, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, betas=betas, expression=expression)
                root_init = output.joints[0][0]
                joint_world = (output.joints[0] + transl).numpy()
                joint_img, joint_cam, focal, princpt = project_2d(data, i, j, joint_world)
                joint_world, joint_cam, joint_img = torch.FloatTensor(joint_world), torch.FloatTensor(joint_cam), torch.FloatTensor(joint_img)
                RTs = corresponding_points_alignment(joint_world[None], joint_cam[None])
                R = RTs.R.permute(0,2,1)[0]
                t = RTs.T[0]

                # apply rigid transformation
                global_orient = axis_angle_to_matrix(global_orient).view(3,3)
                global_orient = torch.matmul(R, global_orient)
                global_orient = matrix_to_axis_angle(global_orient).view(1,3)
                transl = -root_init + torch.matmul(R, (root_init + transl).view(3,1)).view(3) + t
                transl = transl.view(1,3)
                
                """
                # for debug
                with torch.no_grad():
                    output = smplx_layer[gender](global_orient=global_orient, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, betas=betas, expression=expression, transl=transl)
                joint_cam_debug = output.joints[0]
                x = joint_cam_debug[:,0] / joint_cam_debug[:,2] * focal[0] + princpt[0]
                y = joint_cam_debug[:,1] / joint_cam_debug[:,2] * focal[1] + princpt[1]
                joint_img_debug = torch.stack((x,y),1)
                print('joint_cam', torch.sqrt(torch.sum((joint_cam - joint_cam_debug)**2,1)).sum(), ' meter')
                print('joint_img', torch.sqrt(torch.sum((joint_img - joint_img_debug)**2,1)).sum(), ' pixel')
                """

                # save smplx parameters in camera coordinate system
                global_orient = global_orient.view(-1).tolist()
                body_pose = body_pose.view(-1).tolist()
                jaw_pose = jaw_pose.view(-1).tolist()
                leye_pose = leye_pose.view(-1).tolist()
                reye_pose = reye_pose.view(-1).tolist()
                left_hand_pose = left_hand_pose.view(-1).tolist()
                right_hand_pose = right_hand_pose.view(-1).tolist()
                betas = betas.view(-1).tolist()
                expression = expression.view(-1).tolist()
                transl = transl.view(-1).tolist()
                save_path = osp.join(dataset_path, smplx_params_path, img_folder_name, data['imgPath'][i][:-4])
                os.makedirs(save_path, exist_ok=True)
                with open(osp.join(save_path, str(j) + '.json'), 'w') as f:
                    json.dump({'global_orient': global_orient, 'body_pose': body_pose, 'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose, 'betas': betas, 'expression': expression, 'transl': transl}, f)
                
                ann_id += 1

            # save camera intrinsic (camera parameters from agora2coco_smpl.py and agora2coco_smplx.py are the same)
            save_path = osp.join(dataset_path, cam_params_path, img_folder_name)
            os.makedirs(save_path, exist_ok=True)
            with open(osp.join(save_path, data['imgPath'][i][:-4] + '.json'), 'w') as f:
                json.dump({'focal': focal, 'princpt': princpt}, f)
            image_id += 1

    with open(osp.join(dataset_path, 'AGORA_' + split + '_SMPLX.json'), 'w') as f:
        json.dump({'images': images, 'annotations': annotations}, f)
        
