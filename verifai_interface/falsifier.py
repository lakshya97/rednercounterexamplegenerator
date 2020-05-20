from redner_adv import *
import torch
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
import argparse
import os

from verifai.features.features import *
from verifai.samplers.feature_sampler import *
from verifai.falsifier import generic_falsifier
from verifai.monitor import specification_monitor

import numpy as np
from dotmap import DotMap

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='The shapenet/imagenet ID of the class', required=True)
parser.add_argument('--hashcode_file', type=str, help='A text file with a list of shapenet hashcodes', required=True)
parser.add_argument('--label', type=int, help='The label corresponding with your shapenet id', required=True)
parser.add_argument('--pose', type=str, choices=['forward', 'top', 'left', 'right'], default='forward')
parser.add_argument('--max_iters', type=int, default=5)

args = parser.parse_args()

NUM_CLASSES = 12
BACKGROUND = 'lighting/blue_white.png'
IMAGENET_FILENAME = 'class_labels.json'
VGG_PARAMS = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}

# For now we hardcode the object id and true label 0-11
OBJ_ID = args.id
LABEL = args.label
POSE = args.pose

hashcode_file = open(args.hashcode_file, 'r')
hashcodes = []
for line in hashcode_file.readlines():
    hashcodes += [line.strip("\n")]

MAX_ITERS = 5
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096

falsifier_params = DotMap()
falsifier_params.n_iters = args.max_iters
falsifier_params.compute_error_table = False
falsifier_params.save_error_table = False
falsifier_params.save_safe_table = False
falsifier_params.fal_thres = 0.5

class confidence_spec(specification_monitor):
    def __init__(self):
        def specification(traj):
            return bool(traj['pred'] == LABEL)
        super().__init__(specification)

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

total_misclassif = 0

falsifier = None
for hashcode in hashcodes:
    obj_filename  = '../ShapeNetCore.v2/' + OBJ_ID + '/' + hashcode + '/models/model_normalized.obj'
    _, mesh_list, _ = pyredner.load_obj(obj_filename)

    features = {'euler_delta': Feature(Box([-.3, .3]))}
    for i, name_mesh in enumerate(mesh_list):
        _, mesh = name_mesh
        features['mesh' + str(i)] = Feature(Array(Box((-.005, .005)), tuple(mesh.vertices.shape)))

    features['obj_id'] = Feature(Constant(OBJ_ID))
    features['hashcode'] = Feature(Constant(hashcode))
    features['pose'] = Feature(Constant(POSE))

    space = FeatureSpace(features)
    halton_params = DotMap()
    halton_params.sample_index = 0 
    halton_params.bases_skipped = 0
    sampler = FeatureSampler.haltonSamplerFor(space, halton_params)
    falsifier = generic_falsifier(sampler=sampler, server_options=server_options,
                monitor=confidence_spec(), falsifier_params=falsifier_params)
    rhos = falsifier.run_falsifier()
    print(rhos)
    misclassif = np.sum(np.invert(rhos))
    print('Number of misclassifications:', misclassif)
    total_misclassif += misclassif

print('Total misclassification rate:', total_misclassif / (len(hashcodes) * MAX_ITERS))


