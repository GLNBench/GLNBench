import torch
import torch.nn.functional as F
import torch.optim as optim

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('SheafNN_Helper')
class SheafNNHelper(MethodHelper):

    def setup(self, backbone_model, data, config, device, init_data):

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

    def train_step_batched(self, state: dict, loaders, data, epoch: int) -> dict:
        """Mini-batch training step.

        Only called when ``supports_batched_training()`` is True and
        ``training.batch_size`` is set.

        Default implementation iterates ``loaders.train_loader`` and calls the
        primary model with CE loss on seed (target) nodes.  Override for
        non-trivial training logic.

        Args:
            state: Method state dict.
            loaders: ``GraphLoaders`` from ``util.graph_sampling``.
            data: Full PyG Data object (for reference / masks).
            epoch: Current epoch (0-based).

        Returns:
            Dict with at least ``'train_loss': float``.
        """
        from util.graph_sampling import get_seed_indices

        model = state['model']
        optimizer = state['optimizer']
        device = state.get('device', next(model.parameters()).device)

        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loaders.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(batch)
            n_seed = get_seed_indices(batch, loaders.sampler_type)

            # Loss on seed/target nodes only
            seed_mask = batch.train_mask[:n_seed]
            idx = seed_mask.nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            loss = F.cross_entropy(out[idx], batch.y[idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'train_loss': total_loss / max(n_batches, 1)}

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
        model = state['model']
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