import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform, skimage.io, skimage.color

from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# skimage version: 0.14.5
# numpy version: 1.16.5
# torch version: 1.3.1
print "skimage version:", skimage.__version__
print "numpy version:", np.__version__
print "torch version:", torch.__version__

def plot_stats(layers, stats, y_label, fname=None):
    plt.figure(figsize=(7,5))
    plt.plot(stats, marker='o')
    plt.xticks(np.arange(len(layers)), layers, rotation=45)
    plt.ylabel(y_label)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(FIGURE_DIR + fname, format='pdf')

def image_loader(device, image_name):
    image = skimage.io.imread(image_name)
    a = skimage.transform.resize(image, (224,224), preserve_range=True)
    a = np.copy(a).astype('uint8')

    # Convert ro RGB if grayscale
    if a.ndim == 2:
        a = skimage.color.gray2rgb(a)

    loader = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = loader(a)
    image = Variable(image, requires_grad=False)
    return image.to(device)

def load_model(model, pretrained=True):
    if model == "vgg19":
        return models.vgg19(pretrained=pretrained)
    elif model == "alexnet":
        return models.alexnet(pretrained=pretrained)
    else:
        assert 0

def compute_statistics(feats):
    # Number of features
    feats = np.reshape(feats, [feats.shape[0], -1])
    num_feats = feats.shape[1]

    # Activations
    prop_m = np.mean(np.mean(np.abs(feats)<=1e-20, axis=1))

    return num_feats, prop_m

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    print "Device:", DEVICE

    import os
    im = os.environ["GROUP_HOME"] + "/datasets/tiny-imagenet-200/val/images/val_0.JPEG"
    a = image_loader(DEVICE, im)
    print a.size()

