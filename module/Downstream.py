import torch
import torch.nn as nn
from module.GTCMol import GTCMol


class GTCMolClassifier(nn.Module):
    def __init__(self, config, n_classes=2, pretrained_path=None):
        """
        config: dict with hyperparameters
        n_classes: number of downstream classes
        pretrained_path: path to pretrained weights
        """
        super(GTCMolClassifier, self).__init__()

        self.encoder = GTCMol(
            input_dim=config.get('input_dim', 147),  
            hidden_dim=config.get('hidden_dim', 256),  
            edge_feature_dim=config.get('edge_feature_dim', 5),
            gnn_layer_num=config.get('gnn_layer_num', 3),

            transformer_layers=config.get('t_n_layers', 4),
            transformer_heads=config.get('t_num_heads', 8),

            dropout_rate=config.get('t_dropout_rate', 0.1),
            att_dropout_rate=config.get('t_attention_dropout_rate', 0.1),

            tau=config.get('tau', 0.1)
        )

        if pretrained_path:
            print(f" Loading pre-trained model: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)  

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  
                else:
                    new_state_dict[k] = v

            try:
                msg = self.encoder.load_state_dict(new_state_dict, strict=False)
                print(f" Weights loaded! Missing keys: {msg.missing_keys}")
            except Exception as e:
                print(f" Weight loading failed: {e}")
                print("Check hidden_dim and input_dim match pre-training!")

        hidden_dim = config.get('hidden_dim', 256)
        cls_dropout = config.get('class_dropoutrate', 0.2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, data):
        mol_vec, _ = self.encoder(data)

        logits = self.classifier(mol_vec)
        return logits