import torch
import torch.nn.functional as F
import torch.optim as optim

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('SheafNN_Helper')
class SheafNNHelper(MethodHelper):

    def setup(self, backbone_model, data, config, device, init_data):
        
        with torch.no_grad():
            backbone_model.eval()
            backbone_model(data)
            backbone_model.train()

        optimizer = optim.Adam(
            backbone_model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 5e-4),
        )
        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'model': backbone_model,
            'optimizer': optimizer,
        }

    def train_step(self, state, data, epoch):
        model = state['model']
        optimizer = state['optimizer']

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}

    def compute_val_loss(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            out = model(data)
            return F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()

    def get_predictions(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            return model(data).argmax(dim=1)

    def get_probabilities(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return F.softmax(model(data), dim=1)
        
    def get_embeddings(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)

    # Optional: enable mini-batch training
    def supports_batched_training(self):
        return True  # default is False