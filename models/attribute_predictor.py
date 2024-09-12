import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder, dropout_rate=0):
        super().__init__()
        self.image_encoder = image_encoder
        self.attribute_sizes = attribute_sizes
        self.dropout = nn.Dropout(dropout_rate)

        self.attribute_predictors = nn.Linear(image_encoder_output_dim, sum(attribute_sizes))
        nn.init.kaiming_normal_(self.attribute_predictors.weight, nonlinearity='relu')
        nn.init.zeros_(self.attribute_predictors.bias)

    def predict_from_features(self, x):
        # if self.batch_predictors:
        outputs = self.attribute_predictors(x)
        outputs = list(torch.split(outputs, self.attribute_sizes, dim=1))
        # else:
        #     outputs = [predictor(x) for predictor in self.attribute_predictors]
        return outputs

    def extract_features(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features
        return x

    def forward(self, x, apply_softmax=False):
        x = self.extract_features(x)
        #TODO: modify dropout layer place
        x = self.dropout(x) # Apply dropout
        outputs = self.predict_from_features(x)
        if apply_softmax:
            outputs = [F.softmax(output, dim=1) for output in outputs]
        return outputs