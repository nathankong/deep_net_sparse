class ModelInfo(object):
    def __init__(self):
        pass

    def get_feature_layer_index_dictionary(self):
        # i.e. convolutional layer features
        raise NotImplementedError

    def get_classifier_layer_index_dictionary(self):
        # i.e. fully connected layer features
        raise NotImplementedError

    def get_layers(self):
        raise NotImplementedError

class VGG19Info(ModelInfo):
    def __init__(self):
        super(VGG19Info, self).__init__()

        self.layers = [
            "conv1_1", "conv1_2", "pool1", 
            "conv2_1", "conv2_2", "pool2", 
            "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3", 
            "conv4_1", "conv4_2", "conv4_3", "conv4_4", "pool4", 
            "conv5_1", "conv5_2", "conv5_3", "conv5_4", "pool5", 
            "fc1", "fc2"
        ]

        self.feature_layer_dict = {
            0: 'conv1_1',
            2: 'conv1_2',
            4: 'pool1',
            5: 'conv2_1',
            7: 'conv2_2',
            9: 'pool2',
            10: 'conv3_1',
            12: 'conv3_2',
            14: 'conv3_3',
            16: 'conv3_4',
            18: 'pool3',
            19: 'conv4_1',
            21: 'conv4_2',
            23: 'conv4_3',
            25: 'conv4_4',
            27: 'pool4',
            28: 'conv5_1',
            30: 'conv5_2',
            32: 'conv5_3',
            34: 'conv5_4',
            36: 'pool5'
        }
        self.classifier_layer_dict = {
            0: 'fc1',
            3: 'fc2',
            6: 'fc3'
        }

    def get_feature_layer_index_dictionary(self):
        return self.feature_layer_dict

    def get_classifier_layer_index_dictionary(self):
        return self.classifier_layer_dict

    def get_layers(self):
        return self.layers


class AlexNetInfo(ModelInfo):
    def __init__(self):
        super(AlexNetInfo, self).__init__()

        # Layers including ReLU modules
#        self.layers = [
#            "conv1", "relu1", "pool1",
#            "conv2", "relu2", "pool2",
#            "conv3", "relu3",
#            "conv4", "relu4",
#            "conv5", "relu5", "pool5",
#            "fc1", "relufc1",
#            "fc2", "relufc2",
#            "fc3"
#        ]

        # Layers without ReLU modules
        self.layers = [
            "conv1", "pool1",
            "conv2", "pool2",
            "conv3",
            "conv4",
            "conv5", "pool5",
            "fc1", "fc2", "fc3"
        ]

        self.feature_layer_dict = {
            0: 'conv1',
            #1: 'relu1',
            2: 'pool1',
            3: 'conv2',
            #4: 'relu2',
            5: 'pool2',
            6: 'conv3',
            #7: 'relu3',
            8: 'conv4',
            #9: 'relu4',
            10: 'conv5',
            #11: 'relu5',
            12: 'pool5'
        }
        self.classifier_layer_dict = {
            1: 'fc1',
            #2: 'relufc1',
            4: 'fc2',
            #5: 'relufc2',
            6: 'fc3'
        }

    def get_feature_layer_index_dictionary(self):
        return self.feature_layer_dict

    def get_classifier_layer_index_dictionary(self):
        return self.classifier_layer_dict

    def get_layers(self):
        return self.layers

# Function to use to get model information object
def get_model_info(model_name):
    if model_name == "vgg19":
        model = VGG19Info()
    elif model_name == "alexnet":
        model = AlexNetInfo()
    else:
        assert 0, "Model information not implemented."
    return model



