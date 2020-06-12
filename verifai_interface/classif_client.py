from redner_adv import *
import torch
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
import argparse
import os
import time
import datetime

import numpy as np
from dotmap import DotMap

from verifai.client import Client

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='Image output directory.', default='verifai_out')

args = parser.parse_args()

NUM_CLASSES = 12
OUT_DIR = args.out
BACKGROUND = 'lighting/blue_white.png'
IMAGENET_FILENAME = 'class_labels.json'
VGG_PARAMS = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}


class Classifier(Client):
    def __init__(self, classifier_data):
        port = classifier_data.port
        bufsize = classifier_data.bufsize
        super().__init__(port, bufsize)

        # PREPROCESSING CODE THAT SHAPES THE NETWORK TO OUR SHAPENET DATASET #
        self.vgg16 = vgg.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

        if not torch.cuda.is_available():
            self.vgg16.load_state_dict(torch.load('torch_models/model_ft.pt', map_location=lambda storage, location: storage))
        else:
            self.vgg16.load_state_dict(torch.load('torch_models/model_ft.pt'))

        self.cur_hashcode = None
        self.iters = 0

    def simulate(self, sample):
        print('The time is', datetime.datetime.now())
        # For filename purposes, we reset self.iters to 0 with each new hashcode
        if not self.cur_hashcode or self.cur_hashcode != sample.hashcode:
            self.cur_hashcode = sample.hashcode
            self.iters = 0
        obj_filename = '../ShapeNetCore.v2/' + sample.obj_id + '/' + sample.hashcode + '/models/model_normalized.obj'
        with torch.no_grad():
            v = SemanticPerturbations(self.vgg16, obj_filename, dims=(224,224), label_names=get_label_names(IMAGENET_FILENAME), 
                                            normalize_params=VGG_PARAMS, background=BACKGROUND, pose=sample.pose, num_classes=NUM_CLASSES, attack_type='benign')
            v.euler_angles += torch.tensor(sample.euler_delta, device=pyredner.get_device())
            print('eulers added')
            for i in range(len(v.shapes)):
                v_shape = v.shapes[i].vertices.shape
                v.shapes[i].vertices += torch.tensor(sample._asdict()['mesh' + str(i)], device=pyredner.get_device()).reshape(v_shape)
            print('vertices added')
            img = v.render_image(out_dir=OUT_DIR + '/' + sample.obj_id, 
                    filename=sample.hashcode + '_' + sample.pose + '_' + str(self.iters) + '.png', no_grad=True)
            print('img rendered')
            pred, out = v.classify(img)
            print('forward pass done')
            res = {}
            res['pred'] = pred
            print(res)
            self.iters += 1
            return res

if __name__ == '__main__':
    PORT = 8888
    BUFSIZE = 4096 * 4

    classifier_data = DotMap()
    classifier_data.port = PORT
    classifier_data.bufsize = BUFSIZE
    client_task = Classifier(classifier_data)
    last_success = time.time()
    while True:
        try:
            client_task.run_client()
            last_success = time.time()
        except RuntimeError as e:
            time.sleep(1)
            # We assume if the falsifier hasn't sent anything in 60s, we are done.
            if time.time() - last_success > 60 * 60:
                print('End of all samples.')
                break
    
