import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle
import os
os.environ["TORCH_HOME"] = "/mnt/fs5/nclkong/deep_models/"

import numpy as np
import collections
from functools import partial

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder

from ModelInfo import get_model_info
from utils import plot_stats, image_loader, load_model, compute_statistics

torch.manual_seed(0)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")
print "Device:", DEVICE

FIGURE_DIR = "./figures/"
RESULTS_DIR = "./results/"

def main(model_name, images_dir):
    # Load model
    m_info = get_model_info(model_name)
    feature_layer_dict = m_info.get_feature_layer_index_dictionary()
    classifier_layer_dict = m_info.get_classifier_layer_index_dictionary()
    layers_order = m_info.get_layers()
    m = load_model(model_name).to(DEVICE)

    # A dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        #print name, out.cpu().size()
        activations[name].append(np.copy(out.cpu().detach().numpy()))

    # Get Conv2d/Pooling layer activations
    for i, module in enumerate(m.features):
    	#if type(module)==nn.Conv2d or type(module)==nn.MaxPool2d:
    	if type(module)==nn.Conv2d or type(module)==nn.ReLU or type(module)==nn.MaxPool2d:
    	#if type(module)==nn.ReLU or type(module)==nn.MaxPool2d:
            name = feature_layer_dict[i]
    	    module.register_forward_hook(partial(save_activation, name))
            #print i, module

    # Get FC layer activations
    for i, module in enumerate(m.classifier):
    	if type(module)==nn.Linear:
            name = classifier_layer_dict[i]
    	    module.register_forward_hook(partial(save_activation, name))
            #print i, module

    dataset = ImageFolder(images_dir, loader=partial(image_loader, DEVICE))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    m.eval()
    for step, (images,y) in enumerate(data_loader):
        # TODO: DEBUG, use 1/10 of the batches for now, for runtime
        if (step+1) % 1 == 0:
            print "Batch {}".format(step+1)
            _ = m(images)

    activations = {name: np.concatenate(outputs, axis=0) for name, outputs in activations.items()}

    n_feats_all = list()
    n_zero_mean_all = list()
    layer_feature_stds = list()
    for layer in layers_order:
        features = activations[layer]
        n_feats, n_zero_mean = compute_statistics(features)
        n_feats_all.append(n_feats)
        n_zero_mean_all.append(n_zero_mean)
        layer_feature_stds.append(features.std(axis=0))

    # Save statistics
    results = dict()
    results["layers"] = layers_order
    results["statistics"] = dict()
    results["statistics"]["zero_mean_proportion"] = n_zero_mean_all
    results["statistics"]["num_features"] = n_feats_all
    results["statistics"]["feature_stds"] = layer_feature_stds
    results_fname = "{}_stats.pkl".format(model_name)
    pickle.dump(results, open(RESULTS_DIR + results_fname, "wb"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vgg19")
    parser.add_argument('--imagedir', type=str, default="./images/")
    args = parser.parse_args()

    model_name = args.model.lower()
    images_dir = args.imagedir.lower()
    print "Model:", model_name
    print "Image directory:", images_dir
    main(model_name, images_dir)


