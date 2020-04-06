import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
from torch.autograd import Variable
import pyredner
import matplotlib.pyplot as plt
import urllib
import zipfile
import requests
import json
import numpy as np
import argparse
import torch.nn as nn

NUM_CLASSES=12
vgg16 = vgg.vgg16(pretrained=True)
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
vgg16.load_state_dict(torch.load('torch_models/model_ft.pt'))

def set_grad(var):
    def hook(grad):
        grad[grad != grad] = 0
        var.grad = grad
    return hook

#the below method was sampled, with approval, from Rehan Durrani's work at https://github.com/harbor-ml/modelzoo/
def get_label_names(filename):
    with open(filename, 'r') as f:
        detect_labels = {
            int(key): value for (key, value) in json.load(f).items()
        }
    return detect_labels

class SemanticPerturbations:
    def __init__(self, framework, filename, dims, label_names, normalize_params, background, pose):
        self.framework = framework
        self.image_dims = dims
        self.label_names = label_names
        self.framework_params = normalize_params
        
        # self.objects = pyredner.load_obj(filename, return_objects=True)
        self.material_map, mesh_list, self.light_map = pyredner.load_obj(filename)
        for _, mesh in mesh_list:
            mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)
        vertices = []
        for _, mesh in mesh_list:
            vertices.append(mesh.vertices)
            mesh.vertices = Variable(mesh.vertices, requires_grad=True)
            mesh.vertices.retain_grad()

        material_id_map = {}
        self.materials = []
        count = 0
        for key, value in self.material_map.items():
            material_id_map[key] = count
            count += 1
            self.materials.append(value)
        
        self.shapes = []
        for mtl_name, mesh in mesh_list:
            #assert(mesh.normal_indices is None)
            self.shapes.append(pyredner.Shape(\
                vertices = mesh.vertices,
                indices = mesh.indices,
                material_id = material_id_map[mtl_name],
                uvs = mesh.uvs,
                normals = mesh.normals,
                uv_indices = mesh.uv_indices))


        self.camera = pyredner.automatic_camera_placement(self.shapes, resolution=(512,512))
        # Compute the center of the teapot
        self.center = torch.mean(torch.cat(vertices), 0)
        self.translation = torch.tensor([0., 0., 0.], device = pyredner.get_device(), requires_grad=True)

        if pose == 'forward':
            self.euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device(), requires_grad=True)
        elif pose == 'top':
            self.euler_angles = torch.tensor([0.35, 0., 0.], device = pyredner.get_device(), requires_grad=True)
        elif pose == 'left':
            self.euler_angles = torch.tensor([0., 0.50, 0.], device = pyredner.get_device(), requires_grad=True)
        elif pose == 'right':
            self.euler_angles = torch.tensor([0., -0.50, 0.], device = pyredner.get_device(), requires_grad=True)

        self.light = pyredner.PointLight(position = (self.camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                                                intensity = torch.tensor((20000.0, 30000.0, 20000.0), device = pyredner.get_device()))
        
        background = pyredner.imread(background)
        self.background = background.to(pyredner.get_device())
        
    # image: the torch variable holding the image
    # net_out: the output of the framework on the image
    # label: an image label (given as an integer index)
    # returns: the gradient of the image w.r.t the given label
    def _get_gradients(self, image, net_out, label):
        score = net_out[0][label]
        score.backward(retain_graph=True)
        #return image.grad

    # classifies the input image 
    # image: np array of input image
    # label: correct class label for image
    def classify(self, image):
        self.framework.eval()
        #transform image before classifying by standardizing values
        mean, std = self.framework_params["mean"], self.framework_params["std"]
        normalize = transforms.Normalize(mean, std)
        image = normalize(image.cpu()[0])
        image = image.unsqueeze(0)

        #forward pass
        fwd = self.framework.forward(image)
        
        #classification via softmax
        probs, top5 = torch.topk(fwd, 5, 1, True, True)
        top5 = top5[0]
        probs = probs[0]
        #probs = torch.nn.functional.softmax(fwd[0], dim=0).data.numpy()
        #top3 = np.argsort(probs)[-3:][::-1]
        labels = [(self.label_names[label.item()], probs[idx].item()) for idx, label in enumerate(top5)]
        print("Top 5: ", labels)
        prediction_idx = top5[0]
        
        #prediction_idx = int(torch.argmax(fwd[0]))
        #prediction = self.label_names[prediction_idx] 
        return prediction_idx, fwd

    # You might need to combine the detector and the renderer into one class...this will enable you to retrieve gradients of the placement w.r.t the input stuff

    # model the scene based on current instance params
    def _model(self):
        # Get the rotation matrix from Euler angles
        rotation_matrix = pyredner.gen_rotate_matrix(self.euler_angles)
        self.euler_angles.retain_grad()
        # Shift the vertices to the center, apply rotation matrix,
        # shift back to the original space, then apply the translation.
        vertices = []
        for shape in self.shapes:
            shape_v = shape.vertices.clone().detach()
            shape.vertices = (shape_v - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
            shape.vertices.retain_grad()
            shape.vertices.register_hook(set_grad(shape.vertices))
            shape.normals = pyredner.compute_vertex_normal(shape.vertices, shape.indices)
            vertices.append(shape.vertices.clone().detach())
        self.center = torch.mean(torch.cat(vertices), 0)
        # Assemble the 3D scene.
        scene = pyredner.Scene(camera=self.camera, shapes=self.shapes, materials=self.materials)
        # Render the scene.
        img = pyredner.render_deferred(scene, lights=[self.light], alpha=True)
        
        return img

    # render the image properly and downsample it to the right dimensions
    def render_image(self, out_dir=None, filename=None):
        if (out_dir is None) is not (filename is None):
            raise Exception("must provide both out dir and filename if you wish to save the image")

        dummy_img = self._model()

        #honestly dont know if this makes a difference, but...
        self.euler_angles.data = torch.tensor([0., 0., 0.], device = pyredner.get_device(), requires_grad=True)
        img = self._model()
        #just meant to prevent rotations from being stacked onto one another with the above line

        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + self.background * (1 - alpha)

        # Visualize the initial guess
        eps = 1e-6
        img = torch.pow(img + eps, 1.0/2.2) # add .data to stop PyTorch from complaining
        img = torch.nn.functional.interpolate(img.T.unsqueeze(0), size=self.image_dims, mode='bilinear')
        img.retain_grad()
        
        #save image
        if out_dir is not None and filename is not None:
            plt.imsave(out_dir + "/" + filename, img[0].T.data.cpu().numpy())
        
        return img.permute(0,1,3,2)

    # does a gradient attack on the image to induce misclassification. if you want to move away from a specific class
    # then subtract. else, if you want to move towards a specific class, then add the gradient instead.
    def attack_FGSM(self, label, out_dir=None, save_title=None, steps=5, vertex_eps=0.001, pose_eps=0.05, vertex_attack=True, pose_attack=True):
        if out_dir is not None and save_title is None:
            raise Exception("Must provide image title if out dir is provided")
        elif save_title is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")
        
        if save_title is not None:
            filename = save_title + ".png"
        else:
            filename = save_title

        # classify 
        img = self.render_image(out_dir=out_dir, filename=filename) 
        # only there to zero out gradients.
        optimizer = torch.optim.Adam([self.translation, self.euler_angles], lr=0) 
        print("CLASSIFYING BENIGN")
        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)
            # get gradients
            self._get_gradients(img.cpu(), net_out, label)
            delta = 1e-6
            inf_count = 0
            nan_count = 0

            #attack each shape's vertices
            if vertex_attack:
                for shape in self.shapes:
                    if not torch.isfinite(shape.vertices.grad).all():
                        inf_count += 1
                    elif torch.isnan(shape.vertices.grad).any():
                        nan_count += 1
                    else:
                        #subtract because we are trying to decrease the classification score of the label
                        shape.vertices -= torch.sign(shape.vertices.grad/(torch.norm(shape.vertices.grad) + delta)) * vertex_eps
            
            #self.translation = self.translation - torch.sign(self.translation.grad/torch.norm(self.translation.grad) + delta) * eps
            #self.translation.retain_grad()
            #print(self.euler_angles)
            if pose_attack:
                self.euler_angles.data -= torch.sign(self.euler_angles.grad/(torch.norm(self.euler_angles.grad) + delta)) * pose_eps
            
            #print("rotation grad: ", self.euler_angles.grad)
            #optimizer.step()
            
            if save_title is not None:
                filename = save_title + "_iter_" + str(i) + ".png"
            else:
                filename = save_title

            img = self.render_image(out_dir=out_dir, filename=filename)

        final_pred, net_out = self.classify(img)
        return final_pred, img

    # does a gradient attack on the image to induce misclassification. if you want to move away from a specific class
    # then subtract. else, if you want to move towards a specific class, then add the gradient instead.
    def attack_PGD(self, label, out_dir=None, save_title=None, steps=5, vertex_epsilon=1.0, pose_epsilon=1.0, vertex_lr=0.001, pose_lr=0.05, 
            vertex_attack=True, pose_attack=True):

        if out_dir is not None and save_title is None:
            raise Exception("Must provide image title if out dir is provided")
        elif save_title is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")
        
        if save_title is not None:
            filename = save_title + ".png"
        else:
            filename = save_title

        # classify
        img = self.render_image(out_dir=out_dir, filename=filename)
        
        # only there to zero out gradients.
        optimizer = torch.optim.Adam([self.translation, self.euler_angles], lr=0)

        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)

            # get gradients
            self._get_gradients(img.cpu(), net_out, label)
            delta = 1e-6
            inf_count = 0
            nan_count = 0
            
            if vertex_attack: 
                #attack each shape's vertices
                for shape in self.shapes:
                    if not torch.isfinite(shape.vertices.grad).all():
                        inf_count += 1
                    elif torch.isnan(shape.vertices.grad).any():
                        nan_count += 1
                    else:
                        #subtract because we are trying to decrease the classification score of the label
                        shape.vertices -= torch.clamp(shape.vertices.grad/(torch.norm(shape.vertices.grad) + delta) * vertex_lr, -vertex_epsilon, vertex_epsilon)

            #self.translation = self.translation - torch.sign(self.translation.grad/torch.norm(self.translation.grad) + delta) * eps
            #self.translation.retain_grad()
            #print(self.euler_angles)
            
            if pose_attack:
                self.euler_angles.data -= torch.clamp(self.euler_angles.grad/(torch.norm(self.euler_angles.grad) + delta) * pose_lr, -pose_epsilon, pose_epsilon)
            
            # self.euler_angles.data -= torch.sign(self.euler_angles.grad/(torch.norm(self.euler_angles.grad) + delta)) * eps
            #print("rotation grad: ", self.euler_angles.grad)
            #optimizer.step()
            if save_title is not None:
                filename = save_title + "_iter_" + str(i) + ".png"
            else:
                filename = save_title

            img = self.render_image(out_dir=out_dir, filename=filename)
        
        final_pred, net_out = self.classify(img)
        return final_pred, img

#######################
#### USAGE EXAMPLE ####
#######################

# #for vgg16, shape is (224,224)
# parser = argparse.ArgumentParser()
# parser.add_argument('--id', type=str)
# parser.add_argument('--hashcode', type=str)
# parser.add_argument('--label', type=int)
# parser.add_argument('--pose', type=str, choices=['forward', 'top', 'left', 'right'])
# parser.add_argument('--attack', type=str, choices=['FGSM'])

# args = parser.parse_args()
# label = args.label
# attack_type = args.attack
# pose = args.pose
# hashcode = args.hashcode
# obj_id = args.id

# background = "lighting/blue_white.png"
# imagenet_filename = "imagenet_labels.json"

# vgg_params = {'mean': torch.tensor([0.485, 0.456, 0.406]), 'std': torch.tensor([0.229, 0.224, 0.225])}
# obj_filename = "/home/lakshya/ShapeNetCore.v2/" + obj_id + "/" + args.hashcode + "/models/model_normalized.obj"

# if attack_type is None:
#     out_dir = "out/benign/" + obj_id 
# else:
#     out_dir = "out/" + attack_type + "/" + obj_id

# #out_dir += "/" + hashcode

# v = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=get_label_names(imagenet_filename), normalize_params=vgg_params, background=background, pose=pose)
# #v.attack_FGSM(label, out_dir)
# v.render_image(out_dir=out_dir, filename=hashcode + '_' + pose + ".png")

# if attack_type == "FGSM":
#     v.attack_FGSM(label, out_dir, filename=hashcode + '_' + pose)


#a note: to insert any other obj detection framework, you must simply load the model in, get the mean/stddev of the data per channel in an image 
#and get the index to label mapping (the last two steps are only needed(if not trained on imagenet, which is provided above),
#now, you have a fully generic library that can read in any .obj file, classify the image, and induce a misclassification through the attack alg
