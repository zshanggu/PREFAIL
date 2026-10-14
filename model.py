"""PREFAIL risk-prediction model.

Four encoders feed a Transformer fusion encoder:
- image encoder:           ImageNet-pretrained ResNet18 shared by both cameras, one feature per frame
- relative-motion encoder: per camera, the motion values plus their frame-to-frame change, with
                           residual 1D convolutions over the window
- robot-state encoder:     separate tokens for end-effector position, joint angles and joint speeds
- past-risk encoder:       one token per past risk value (value, change), aligned in time with the frames
- fusion encoder:          Transformer over all tokens plus a [CLS] token whose output gives the risk
                           (default), or a 1D ResNet over the frames (--fusion resnet_small ... resnet152)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# shared_state layout (per window): EE(3) + joint angles(6) + joint speeds(6) of the current frame,
# followed by the past (window_size - 1) risk values
ROBOT_STATE_SLICES = [(0, 3), (3, 9), (9, 15)]  # EE, joint angles, joint speeds
N_ROBOT_STATE = 15

# Token types (index into MultiModalRiskModel.type_embed)
CAM_MOTION = (0, 1)
CAM_IMAGE = (2, 3)
ROBOT_STATE = (4, 5, 6)
PAST_RISK = 7
CLS = 8
N_TOKEN_TYPES = 9


class PretrainedFrameEncoder(nn.Module):
    """ImageNet-pretrained ResNet18 trunk applied to every frame.

    Frames are resized to input_size and ImageNet-normalized. BatchNorm statistics stay frozen
    (the trunk's BN layers are always in eval mode), so training and evaluation behave the same.
    """
    def __init__(self, out_dim=128, input_size=112, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(resnet.children())[:-1])  # up to global avgpool → 512
        self.proj = nn.Linear(512, out_dim)
        self.input_size = input_size
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        for m in self.trunk.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        # x: (batch, frames, 3, H, W) in [0, 1] → (batch, frames, out_dim)
        batch_size, frames = x.shape[:2]
        x = x.reshape(batch_size * frames, *x.shape[2:])
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        x = (x - self.img_mean) / self.img_std
        x = self.trunk(x).flatten(1)
        return self.proj(x).view(batch_size, frames, -1)


class TemporalMotionEncoder(nn.Module):
    """Encodes one camera's relative-motion sequence with temporal context.

    Per step: the standardized values plus their frame-to-frame change, projected to d_model,
    then two pre-norm residual 1D convolutions (kernel 3) over the window.
    (batch, C, T) → (batch, T, d_model)
    """
    def __init__(self, in_channels=3, d_model=128):
        super().__init__()
        self.inp = nn.Linear(2 * in_channels, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.convs = nn.ModuleList([nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(2)])

    def forward(self, x):
        dx = torch.diff(x, dim=2, prepend=x[:, :, :1])
        h = self.inp(torch.cat([x, dx], dim=1).transpose(1, 2))
        for norm, conv in zip(self.norms, self.convs):
            h = h + conv(F.gelu(norm(h)).transpose(1, 2)).transpose(1, 2)
        return h


def group_norm(channels, max_groups=8):
    """GroupNorm with up to 8 groups: batch-size independent, so training and one-window evaluation match."""
    groups = max_groups
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Basic 1D residual block (ResNet18/34): two 3-tap convolutions."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = group_norm(out_channels)
        self.downsample = downsample
        nn.init.zeros_(self.norm2.weight)  # block starts as identity; deep stacks otherwise collapse

    def forward(self, x):
        out = self.dropout(F.relu(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class BottleneckBlock(nn.Module):
    """Bottleneck 1D residual block (ResNet50/101/152): 1-tap, 3-tap, 1-tap convolutions."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm1 = group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = group_norm(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.norm3 = group_norm(out_channels * self.expansion)
        self.downsample = downsample
        nn.init.zeros_(self.norm3.weight)  # block starts as identity; deep stacks otherwise collapse

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


# 1D ResNet fusion variants: (block, blocks per stage, channels per stage, temporal stride per stage)
RESNET_FUSIONS = {
    'resnet_small': (ResidualBlock, [2, 2], (64, 128), (1, 1)),
    'resnet18': (ResidualBlock, [2, 2, 2, 2], (64, 128, 256, 512), (1, 2, 2, 2)),
    'resnet34': (ResidualBlock, [3, 4, 6, 3], (64, 128, 256, 512), (1, 2, 2, 2)),
    'resnet50': (BottleneckBlock, [3, 4, 6, 3], (64, 128, 256, 512), (1, 2, 2, 2)),
    'resnet101': (BottleneckBlock, [3, 4, 23, 3], (64, 128, 256, 512), (1, 2, 2, 2)),
    'resnet152': (BottleneckBlock, [3, 8, 36, 3], (64, 128, 256, 512), (1, 2, 2, 2)),
}
FUSIONS = ['transformer'] + list(RESNET_FUSIONS)


class ResNetFusion(nn.Module):
    """1D ResNet over the frames of the window.

    Input: (batch, in_channels, T), one feature vector per frame. A 1x1 convolution projects it to 64
    channels, residual stages follow, then average pooling over time and a linear head.
    """
    def __init__(self, variant, in_channels, dropout=0.3):
        super().__init__()
        block, layers, channels, strides = RESNET_FUSIONS[variant]
        self.proj = nn.Sequential(nn.Conv1d(in_channels, 64, 1), group_norm(64), nn.ReLU(inplace=True))
        stages, width = [], 64
        for out_channels, n_blocks, stride in zip(channels, layers, strides):
            out_width = out_channels * block.expansion
            downsample = None
            if stride != 1 or width != out_width:
                downsample = nn.Sequential(nn.Conv1d(width, out_width, 1, stride=stride, bias=False),
                                           group_norm(out_width))
            blocks = [block(width, out_channels, stride, downsample)]
            blocks += [block(out_width, out_channels) for _ in range(1, n_blocks)]
            stages.append(nn.Sequential(*blocks))
            width = out_width
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stages(self.proj(x))
        return self.head(x.mean(dim=2))


def fusion_from_state_dict(state_dict):
    """Fusion variant of a saved checkpoint, inferred from its parameter names."""
    keys = [k[7:] if k.startswith('module.') else k for k in state_dict]
    if not any(k.startswith('resnet_fusion.') for k in keys):
        return 'transformer'
    n_stages = len({k.split('.')[2] for k in keys if k.startswith('resnet_fusion.stages.')})
    blocks = [len({k.split('.')[3] for k in keys if k.startswith(f'resnet_fusion.stages.{i}.')})
              for i in range(n_stages)]
    bottleneck = any('.conv3.' in k for k in keys if k.startswith('resnet_fusion.'))
    for name, (block, layers, _, _) in RESNET_FUSIONS.items():
        if layers == blocks and (block is BottleneckBlock) == bottleneck:
            return name
    raise ValueError(f"Unknown ResNet fusion layout: {blocks} blocks per stage, bottleneck={bottleneck}")


def _mlp(in_dim, d_model):
    return nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))


class MultiModalRiskModel(nn.Module):
    """
    Inputs (one window):
        states_cam1, states_cam2: (batch, 3, T)   relative motion per camera
        shared_state:             (batch, 19, 1)  current-frame robot state + past risk
        rgb1, rgb2:               (batch, T, 3, H, W) camera frames in [0, 1]
    Tokens: per camera T motion + T image tokens, 3 robot-state tokens, T-1 past-risk tokens, [CLS].
    Learned time embeddings (per frame) and type embeddings (per token kind) are added.

    fusion='transformer' fuses the tokens with a Transformer and reads the risk from [CLS]. A ResNet fusion
    ('resnet_small', 'resnet18', ..., 'resnet152') instead concatenates, per frame, both cameras' motion and image
    features, that frame's past-risk feature (zero for the last frame) and the robot-state features, and runs a
    1D ResNet over the T frames.

    State inputs are standardized with training-set statistics (set_input_stats), stored as buffers
    so they are saved in the checkpoint; past risk stays in [0, 1].
    forward() returns risk probabilities, or logits with return_logits=True (for the loss).
    """
    def __init__(self, window_size=5, cam_state_channels=3, d_model=128, n_heads=4, n_layers=3,
                 dropout=0.1, image_size=112, pretrained=True, fusion='transformer'):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"Unknown fusion {fusion!r}; choose from {FUSIONS}")
        self.window_size = window_size
        self.fusion_type = fusion

        self.register_buffer('cam1_mean', torch.zeros(cam_state_channels))
        self.register_buffer('cam1_std', torch.ones(cam_state_channels))
        self.register_buffer('cam2_mean', torch.zeros(cam_state_channels))
        self.register_buffer('cam2_std', torch.ones(cam_state_channels))
        self.register_buffer('robot_mean', torch.zeros(N_ROBOT_STATE))
        self.register_buffer('robot_std', torch.ones(N_ROBOT_STATE))

        self.image_encoder = PretrainedFrameEncoder(out_dim=d_model, input_size=image_size, pretrained=pretrained)
        self.motion_encoders = nn.ModuleList([TemporalMotionEncoder(cam_state_channels, d_model) for _ in range(2)])
        self.image_norm = nn.LayerNorm(d_model)
        self.motion_norm = nn.LayerNorm(d_model)
        self.robot_state_encoders = nn.ModuleList([_mlp(hi - lo, d_model) for lo, hi in ROBOT_STATE_SLICES])
        self.past_risk_encoder = _mlp(2, d_model)  # (value, change) per past step

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.type_embed = nn.Parameter(torch.zeros(N_TOKEN_TYPES, d_model))
        for p in [self.cls_token, self.time_embed, self.type_embed]:
            nn.init.trunc_normal_(p, std=0.02)

        if fusion == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.fusion = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
        else:
            # per frame: 2 cameras x (motion, image) + past risk + 3 robot-state features
            self.resnet_fusion = ResNetFusion(fusion, in_channels=(4 + 1 + len(ROBOT_STATE_SLICES)) * d_model)

    def set_input_stats(self, cam1_mean, cam1_std, cam2_mean, cam2_std, robot_mean, robot_std):
        """Store training-set mean/std used to standardize the state inputs."""
        for name, value in [('cam1_mean', cam1_mean), ('cam1_std', cam1_std),
                            ('cam2_mean', cam2_mean), ('cam2_std', cam2_std),
                            ('robot_mean', robot_mean), ('robot_std', robot_std)]:
            buf = getattr(self, name)
            buf.copy_(torch.as_tensor(value, dtype=buf.dtype, device=buf.device))

    def pretrained_parameters(self):
        """Parameters initialized from ImageNet; trained with a reduced learning rate."""
        return self.image_encoder.trunk.parameters()

    def forward(self, states_cam1, states_cam2, shared_state, rgb1, rgb2, return_logits=False):
        batch_size, _, time_steps = states_cam1.shape
        states_cam1 = (states_cam1 - self.cam1_mean[None, :, None]) / self.cam1_std[None, :, None]
        states_cam2 = (states_cam2 - self.cam2_mean[None, :, None]) / self.cam2_std[None, :, None]
        shared = shared_state.squeeze(2)
        robot_state = (shared[:, :N_ROBOT_STATE] - self.robot_mean) / self.robot_std
        past_risk = shared[:, N_ROBOT_STATE:]                                   # (batch, T - 1)

        time_embed = self.time_embed[:, :time_steps]
        motion, image = [], []
        for cam, (states, rgb) in enumerate([(states_cam1, rgb1), (states_cam2, rgb2)]):
            motion.append(self.motion_norm(self.motion_encoders[cam](states)))   # (batch, T, d)
            image.append(self.image_norm(self.image_encoder(rgb)))               # (batch, T, d)
        robot = [self.robot_state_encoders[k](robot_state[:, lo:hi])           # (batch, d) each
                 for k, (lo, hi) in enumerate(ROBOT_STATE_SLICES)]
        risk_features = torch.stack([past_risk, torch.diff(past_risk, dim=1, prepend=past_risk[:, :1])], dim=-1)
        risk = self.past_risk_encoder(risk_features)                            # (batch, T - 1, d)

        if self.fusion_type == 'transformer':
            tokens = [(self.cls_token + self.type_embed[CLS]).expand(batch_size, -1, -1)]
            for cam in range(2):
                tokens.append(motion[cam] + time_embed + self.type_embed[CAM_MOTION[cam]])
                tokens.append(image[cam] + time_embed + self.type_embed[CAM_IMAGE[cam]])
            for k in range(len(robot)):
                tokens.append((robot[k] + self.type_embed[ROBOT_STATE[k]]).unsqueeze(1))
            # past risk k is the prediction for the window ending at frame k: it shares frame k's time embedding
            tokens.append(risk + self.time_embed[:, :risk.shape[1]] + self.type_embed[PAST_RISK])
            logits = self.head(self.fusion(torch.cat(tokens, dim=1))[:, 0])
        else:
            # one feature vector per frame; past risk k sits at frame k, the last frame has none
            risk = F.pad(risk, (0, 0, 0, time_steps - risk.shape[1]))
            robot = [r.unsqueeze(1).expand(-1, time_steps, -1) for r in robot]
            frames = torch.cat(motion[:1] + image[:1] + motion[1:] + image[1:] + [risk] + robot, dim=-1)
            logits = self.resnet_fusion(frames.transpose(1, 2))
        return logits if return_logits else torch.sigmoid(logits)


def build_model(window_size=5, pretrained=True, fusion='transformer'):
    """pretrained: load ImageNet weights for the image encoder (False when a checkpoint is loaded next).
    fusion: 'transformer' or a 1D ResNet variant (see FUSIONS)."""
    return MultiModalRiskModel(window_size=window_size, pretrained=pretrained, fusion=fusion)


def load_model(model_path, window_size=5, device=None):
    """Load a trained model checkpoint (a state_dict) for inference."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model = build_model(window_size=window_size, pretrained=False, fusion=fusion_from_state_dict(state_dict))
    model.load_state_dict(state_dict)
    return model.to(device).eval()
