import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ModeloAdaptativoSenas(nn.Module):

    def __init__(self, num_classes, tipos_senas_dict=None, frames_estaticos=8):
        super(ModeloAdaptativoSenas, self).__init__()

        self.num_classes = num_classes
        self.tipos_senas_dict = tipos_senas_dict or {}
        self.frames_estaticos = frames_estaticos

        self.input_size_una_mano = 63
        self.input_size_dos_manos = 126
        self.hidden_size = 256
        self.num_layers = 2

        self.lstm_estatico = nn.LSTM(
            input_size=self.input_size_una_mano,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.lstm_dinamico = nn.LSTM(
            input_size=self.input_size_dos_manos,
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

        self._inicializar_pesos()

        logger.info(f"Modelo LSTM inicializado: {num_classes} clases")
        logger.info(f"  - LSTM Estático: {self.input_size_una_mano} features (1 mano)")
        logger.info(f"  - LSTM Dinámico: {self.input_size_dos_manos} features (2 manos)")
        logger.info(f"  - Frames usados para señas estáticas: {self.frames_estaticos}")

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

    def _seleccionar_mano_valida(self, x):
        batch_size, num_frames, features = x.shape

        mitad_a = x[:, :, :self.input_size_una_mano]
        mitad_b = x[:, :, self.input_size_una_mano:]

        energia_a = mitad_a.abs().sum(dim=(1, 2))
        energia_b = mitad_b.abs().sum(dim=(1, 2))

        usar_b = (energia_b > energia_a).view(batch_size, 1, 1)
        seleccionado = torch.where(usar_b, mitad_b, mitad_a)

        return seleccionado

    def forward_estatica(self, x):
        batch_size, num_frames, features = x.shape

        if not hasattr(self, 'lstm_estatico'):
            raise AttributeError("Modelo no tiene atributo 'lstm_estatico'")

        if features == self.input_size_dos_manos:
            x = self._seleccionar_mano_valida(x)
        elif features != self.input_size_una_mano:
            raise ValueError(f"Input incorrecto: {features} features, esperado {self.input_size_una_mano} o {self.input_size_dos_manos}")

        self._validar_input_size(x, self.input_size_una_mano)

        frames_usar = min(self.frames_estaticos, num_frames)
        start_idx = (num_frames - frames_usar) // 2
        x_input = x[:, start_idx:start_idx + frames_usar]

        x_input = x_input.contiguous()

        lstm_out, (hidden, cell) = self.lstm_estatico(x_input)

        features_finales = lstm_out[:, -1, :]

        return self.classifier_estatico(features_finales)

    def forward_dinamica(self, x):
        batch_size, num_frames, features = x.shape

        if not hasattr(self, 'lstm_dinamico'):
            raise AttributeError("Modelo no tiene atributo 'lstm_dinamico'")

        if features == self.input_size_una_mano:
            padding = torch.zeros(batch_size, num_frames, self.input_size_una_mano, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=2)
        elif features != self.input_size_dos_manos:
            raise ValueError(f"Input incorrecto: {features} features, esperado {self.input_size_una_mano} o {self.input_size_dos_manos}")

        self._validar_input_size(x, self.input_size_dos_manos)

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

        output_final = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)

        tipos_unicos = set(tipos_batch)

        for tipo in tipos_unicos:
            indices = [i for i, t in enumerate(tipos_batch) if t == tipo]
            if not indices:
                continue

            x_subset = x[indices].contiguous()

            if tipo == 'ESTATICA':
                salida = self.forward_estatica(x_subset)
            else:
                salida = self.forward_dinamica(x_subset)

            if salida.dtype != output_final.dtype:
                salida = salida.to(output_final.dtype)

            output_final[indices] = salida

        return output_final

    def forward(self, x, tipos_batch=None):
        if tipos_batch is None:
            tipos_batch = ['DINAMICA'] * x.shape[0]

        return self.forward_batch_mixto(x, tipos_batch)

    def get_tipo_sena(self, nombre_sena):
        if nombre_sena in self.tipos_senas_dict:
            return self.tipos_senas_dict[nombre_sena]

        return 'DINAMICA'


def cargar_modelo_adaptativo(ruta_modelo, device):
    try:
        checkpoint = torch.load(ruta_modelo, map_location=device)

        clases = checkpoint.get('clases', [])
        tipos_senas = checkpoint.get('tipos_senas', {})
        num_clases = len(clases)
        frames_estaticos = checkpoint.get('frames_estaticos', 8)

        modelo = ModeloAdaptativoSenas(num_clases, tipos_senas, frames_estaticos)
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


def contar_parametros(modelo):
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'entrenables': trainable_params,
        'congelados': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 ** 2)
    }