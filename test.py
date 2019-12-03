# Run this to get layer indices for different models
import os
os.environ["TORCH_HOME"] = os.environ["GROUP_HOME"] + "/deep_models/"

import torch
import torchvision.models as models

print "torch version:", torch.__version__

def load_model(model, pretrained=True):
    if model == "vgg19":
        return models.vgg19(pretrained=pretrained)
    elif model == "resnet18":
        return models.resnet18(pretrained=pretrained)
    elif model == "alexnet":
        return models.alexnet(pretrained=pretrained)
    else:
        assert 0

def main():
    m = load_model("alexnet", pretrained=True)
    #print m

    for i, module in enumerate(m.named_modules()):
        print i, module

if __name__ == "__main__":
    main()

