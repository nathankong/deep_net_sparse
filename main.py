import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle
import os
os.environ["TORCH_HOME"] = os.environ["GROUP_HOME"] + "/deep_models/"

import numpy as np
import collections
import skimage.transform, skimage.io
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from ModelInfo import get_model_info

# skimage version: 0.14.5
# numpy version: 1.16.5
# torch version: 1.3.1
print "skimage version:", skimage.__version__
print "numpy version:", np.__version__
print "torch version:", torch.__version__

torch.manual_seed(0)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
print "Device:", DEVICE

FIGURE_DIR = "./figures/"
RESULTS_DIR = "./results/"

def compute_statistics(feats):
    # Number of features
    feats = np.reshape(feats, [feats.shape[0], -1])
    num_feats = feats.shape[1]

    # Activations
    mean_feats = feats.mean(axis=0)
    prop_m = np.sum(mean_feats == 0) / float(mean_feats.shape[0])
    num_zero_mean = np.sum(mean_feats == 0)

    # STD of features
    std_feats = feats.std(axis=0) # std along image dimension
    prop_s = np.sum(std_feats == 0) / float(std_feats.shape[0])

    return num_feats, prop_m, prop_s, num_zero_mean

def plot_stats(layers, stats, y_label, fname=None):
    plt.figure(figsize=(7,5))
    plt.plot(stats, marker='o')
    plt.xticks(np.arange(len(layers)), layers, rotation=45)
    plt.ylabel(y_label)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(FIGURE_DIR + fname, format='pdf')

def image_loader(image_name):
    image = skimage.io.imread(image_name)
    a = skimage.transform.resize(image, (224,224), preserve_range=True)
    a = np.copy(a).astype('uint8')

    loader = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = loader(a)
    image = Variable(image, requires_grad=False)
    return image.to(DEVICE)

def load_model(model, pretrained=True):
    if model == "vgg19":
        return models.vgg19(pretrained=pretrained)
    elif model == "alexnet":
        return models.alexnet(pretrained=pretrained)
    else:
        assert 0

def main(model_name):
    # Load model
    m_info = get_model_info(model_name)
    feature_layer_dict = m_info.get_feature_layer_index_dictionary()
    classifier_layer_dict = m_info.get_classifier_layer_index_dictionary()
    layers_order = m_info.get_layers()
    m = load_model(model_name)

    # A dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        print name, out.cpu().size()
        activations[name].append(out.cpu())

    # Get Conv2d/Pooling layer activations
    for i, module in enumerate(m.features):
    	if type(module)==nn.Conv2d or type(module)==nn.MaxPool2d:
    	    # partial to assign the layer name to each hook
            name = feature_layer_dict[i]
    	    module.register_forward_hook(partial(save_activation, name))
            print i, module

    # Get FC layer activations
    for i, module in enumerate(m.classifier):
    	if type(module)==nn.Linear:
    	    # partial to assign the layer name to each hook
            name = classifier_layer_dict[i]
    	    module.register_forward_hook(partial(save_activation, name))
            print i, module

    dataset = ImageFolder("images/", loader=image_loader)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    m.eval()
    for step, (images,y) in enumerate(data_loader):
        _ = m(images.to(DEVICE))

    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    n_feats_all = list()
    n_zero_mean_all = list()
    n_zero_std_all = list()
    num_zero_mean_all = list()
    for layer in layers_order:
        features = activations[layer]
        features = features.data.cpu().numpy()
        n_feats, n_zero_mean, n_zero_std, num_zero_mean = compute_statistics(features)
        n_feats_all.append(n_feats)
        n_zero_mean_all.append(n_zero_mean)
        n_zero_std_all.append(n_zero_std)
        num_zero_mean_all.append(num_zero_mean)

    # Save statistics
    results = dict()
    results["layers"] = layers_order
    results["statistics"] = dict()
    results["statistics"]["zero_mean_proportion"] = n_zero_mean_all
    results["statistics"]["zero_std_proportion"] = n_zero_std_all
    results["statistics"]["num_features"] = n_feats_all
    results["statistics"]["number_zero_mean"] = num_zero_mean_all
    results_fname = "{}_stats.pkl".format(model_name)
    pickle.dump(results, open(RESULTS_DIR + results_fname, "wb"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

    model_name = args.model.lower()
    main(model_name)


