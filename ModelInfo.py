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
            "conv1_1", "conv1_1_relu", "conv1_2", "conv1_2_relu", "pool1", 
            "conv2_1", "conv2_1_relu", "conv2_1", "conv2_2_relu", "pool2", 
            "conv3_1", "conv3_1_relu", "conv3_2", "conv3_2_relu", "conv3_3", "conv3_3_relu", "conv3_4", "conv3_4_relu", "pool3", 
            "conv4_1", "conv4_1_relu", "conv4_2", "conv4_2_relu", "conv4_3", "conv4_3_relu", "conv4_4", "conv4_4_relu", "pool4", 
            "conv5_1", "conv5_1_relu", "conv5_2", "conv5_2_relu", "conv5_3", "conv5_3_relu", "conv5_4", "conv5_4_relu", "pool5", 
            "fc1", "fc2"
        ]

        self.feature_layer_dict = {
            0: 'conv1_1',
            1: 'conv1_1_relu',
            2: 'conv1_2',
            3: 'conv1_2_relu',
            4: 'pool1',
            5: 'conv2_1',
            6: 'conv2_1_relu',
            7: 'conv2_2',
            8: 'conv2_2_relu',
            9: 'pool2',
            10: 'conv3_1',
            11: 'conv3_1_relu',
            12: 'conv3_2',
            13: 'conv3_2_relu',
            14: 'conv3_3',
            15: 'conv3_3_relu',
            16: 'conv3_4',
            17: 'conv3_4_relu',
            18: 'pool3',
            19: 'conv4_1',
            20: 'conv4_1_relu',
            21: 'conv4_2',
            22: 'conv4_2_relu',
            23: 'conv4_3',
            24: 'conv4_3_relu',
            25: 'conv4_4',
            26: 'conv4_4_relu',
            27: 'pool4',
            28: 'conv5_1',
            29: 'conv5_1_relu',
            30: 'conv5_2',
            31: 'conv5_2_relu',
            32: 'conv5_3',
            33: 'conv5_3_relu',
            34: 'conv5_4',
            35: 'conv5_4_relu',
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

        self.layers = [
            "conv1", "conv1_relu", "pool1",
            "conv2", "conv2_relu", "pool2",
            "conv3", "conv3_relu",
            "conv4", "conv4_relu",
            "conv5", "conv5_relu", "pool5",
            "fc1", "fc2", "fc3"
        ]

        self.feature_layer_dict = {
            0: 'conv1',
            1: 'conv1_relu',
            2: 'pool1',
            3: 'conv2',
            4: 'conv2_relu',
            5: 'pool2',
            6: 'conv3',
            7: 'conv3_relu',
            8: 'conv4',
            9: 'conv4_relu',
            10: 'conv5',
            11: 'conv5_relu',
            12: 'pool5'
        }

        self.classifier_layer_dict = {
            1: 'fc1',
            4: 'fc2',
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



