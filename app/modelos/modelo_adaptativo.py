import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ModeloAdaptativoSenas(nn.Module):
    
    def __init__(self, num_classes, tipos_senas_dict=None):
        super(ModeloAdaptativoSenas, self).__init__()
        
        self.num_classes = num_classes
        self.tipos_senas_dict = tipos_senas_dict or {}
        
        self.input_size_estatico = 63
        self.input_size_dinamico = 126
        self.hidden_size = 256
        self.num_layers = 2
        self.estatico_feat_dim = 256
        self.dinamico_feat_dim = 512
        
        self.lstm_estatico = nn.LSTM(
            input_size=self.input_size_estatico,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        
        self.lstm_dinamico = nn.LSTM(
            input_size=self.input_size_dinamico,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.classifier_estatico = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.classifier_dinamico = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.adaptacion_estatico = nn.Identity()
        self.adaptacion_dinamico = nn.Identity()
        
        self._inicializar_pesos()
        
        logger.info(f"Modelo LSTM inicializado: {num_classes} clases")
        logger.info(f"  - LSTM Estático: {self.input_size_estatico} features (1 mano)")
        logger.info(f"  - LSTM Dinámico: {self.input_size_dinamico} features (2 manos)")

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def _validar_input_size(self, x, esperado):
        batch_size, num_frames, features = x.shape
        if features != esperado:
            raise ValueError(f"Input incorrecto: {features} features, esperado {esperado}")

    def forward_estatica(self, x):
        batch_size, num_frames, features = x.shape
        
        if not hasattr(self, 'lstm_estatico'):
            raise AttributeError("Modelo no tiene atributo 'lstm_estatico'")
        
        if features == self.input_size_dinamico:
            x = x[:, :, :self.input_size_estatico]
        
        self._validar_input_size(x, self.input_size_estatico)
        
        if num_frames <= 4:
            x_input = x
        else:
            frames_middle = min(4, num_frames)
            start_idx = (num_frames - frames_middle) // 2
            x_input = x[:, start_idx:start_idx + frames_middle]
        
        x_input = x_input.contiguous()
        
        lstm_out, (hidden, cell) = self.lstm_estatico(x_input)
        
        features = lstm_out[:, -1, :]
        
        return self.classifier_estatico(features)

    def forward_dinamica(self, x):
        batch_size, num_frames, features = x.shape
        
        if not hasattr(self, 'lstm_dinamico'):
            raise AttributeError("Modelo no tiene atributo 'lstm_dinamico'")
        
        if features == self.input_size_estatico:
            padding = torch.zeros(batch_size, num_frames, self.input_size_estatico, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=2)
        
        self._validar_input_size(x, self.input_size_dinamico)
        
        x = x.contiguous()
        
        lstm_out, (hidden, cell) = self.lstm_dinamico(x)
        
        temporal_weights = self.temporal_attention(lstm_out)
        weighted_features = lstm_out * temporal_weights
        
        temporal_features = torch.mean(weighted_features, dim=1)
        
        return self.classifier_dinamico(temporal_features)

    def forward_batch_mixto(self, x, tipos_batch):
        batch_size = x.shape[0]
        
        if len(x.shape) != 3:
            raise ValueError(f"Forma de tensor incorrecta: {x.shape}, esperado (batch, frames, features)")
        
        outputs_estaticas = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        outputs_dinamicas = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        
        mask_estaticas = [tipo == 'ESTATICA' for tipo in tipos_batch]
        mask_dinamicas = [tipo == 'DINAMICA' for tipo in tipos_batch]
        
        if any(mask_estaticas):
            indices_estaticas = [i for i, mask in enumerate(mask_estaticas) if mask]
            if indices_estaticas:
                x_estaticas = x[indices_estaticas]
                x_estaticas = x_estaticas.contiguous()
                features_estaticas = self.forward_estatica(x_estaticas)
                if features_estaticas.dtype != outputs_estaticas.dtype:
                    features_estaticas = features_estaticas.to(outputs_estaticas.dtype)
                outputs_estaticas[indices_estaticas] = features_estaticas
        
        if any(mask_dinamicas):
            indices_dinamicas = [i for i, mask in enumerate(mask_dinamicas) if mask]
            if indices_dinamicas:
                x_dinamicas = x[indices_dinamicas]
                x_dinamicas = x_dinamicas.contiguous()
                features_dinamicas = self.forward_dinamica(x_dinamicas)
                if features_dinamicas.dtype != outputs_dinamicas.dtype:
                    features_dinamicas = features_dinamicas.to(outputs_dinamicas.dtype)
                outputs_dinamicas[indices_dinamicas] = features_dinamicas
        
        output_final = outputs_estaticas + outputs_dinamicas
        
        return output_final

    def forward(self, x, tipos_batch=None):
        if tipos_batch is None:
            tipos_batch = ['DINAMICA'] * x.shape[0]
        
        return self.forward_batch_mixto(x, tipos_batch)

    def get_tipo_sena(self, nombre_sena):
        if nombre_sena in self.tipos_senas_dict:
            return self.tipos_senas_dict[nombre_sena]
        
        return 'DINAMICA'

def cargar_modelo_adaptativo(ruta_modelo: str, device: torch.device):
    try:
        checkpoint = torch.load(ruta_modelo, map_location=device)
        
        clases = checkpoint.get('clases', [])
        tipos_senas = checkpoint.get('tipos_senas', {})
        num_clases = len(clases)
        
        modelo = ModeloAdaptativoSenas(num_clases, tipos_senas)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.to(device)
        modelo.eval()
        
        metadata = {
            'clases': clases,
            'num_clases': num_clases,
            'tipos_senas': tipos_senas,
            'accuracy': checkpoint.get('accuracy', 0.0),
            'num_frames': checkpoint.get('num_frames', 20),
            'arquitectura': checkpoint.get('arquitectura', 'ModeloAdaptativoSenas'),
            'fecha_entrenamiento': checkpoint.get('fecha_entrenamiento', 'desconocido')
        }
        
        logger.info(f"Modelo cargado: {num_clases} clases, accuracy={metadata['accuracy']:.4f}")
        
        return modelo, metadata
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

def contar_parametros(modelo: nn.Module) -> dict:
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'entrenables': trainable_params,
        'congelados': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 ** 2)
    }

if __name__ == "__main__":
    tipos_test = {
        'A': 'ESTATICA',
        'C': 'ESTATICA',
        'HOLA': 'DINAMICA'
    }
    
    modelo = ModeloAdaptativoSenas(num_classes=3, tipos_senas=tipos_test)
    
    x_estatico = torch.randn(2, 20, 63)
    x_dinamico = torch.randn(2, 20, 126)
    tipos_batch = ['ESTATICA', 'DINAMICA']
    
    logits_estatico = modelo.forward_estatica(x_estatico)
    logits_dinamico = modelo.forward_dinamica(x_dinamico)
    
    print(f"Input estático: {x_estatico.shape} -> Output: {logits_estatico.shape}")
    print(f"Input dinámico: {x_dinamico.shape} -> Output: {logits_dinamico.shape}")
    
    stats = contar_parametros(modelo)
    print(f"\nEstadísticas del modelo:")
    print(f"  Total parámetros: {stats['total']:,}")
    print(f"  Entrenables: {stats['entrenables']:,}")
    print(f"  Tamaño: {stats['total_mb']:.2f} MB")