import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

from .. import config

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DINOv2MixEmbedder:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/dinov2",
            config.DINOV2_MODEL,
            pretrained=True,
            trust_repo=True,
            skip_validation=True,
        )

        self.model.eval().to(self.device)
        self.model = torch.compile(self.model)

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> list[float]:
        tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)

        features = self.model.forward_features(tensor)
        cls = features["x_norm_clstoken"]
        patches = features["x_norm_patchtokens"]
        patch_gem = patches.clamp(min=1e-6).pow(config.P_COEF).mean(dim=1).pow(1.0 / config.P_COEF)
        vector = torch.cat([cls, patch_gem], dim=-1).squeeze(0)
        return vector.float().cpu().tolist()

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([_TRANSFORM(img.convert("RGB")) for img in images]).to(self.device)

        features = self.model.forward_features(batch)
        cls = features["x_norm_clstoken"]
        patches = features["x_norm_patchtokens"]
        patch_gem = patches.clamp(min=1e-6).pow(config.P_COEF).mean(dim=1).pow(1.0 / config.P_COEF)
        vectors = torch.cat([cls, patch_gem], dim=-1)
        return vectors.float().cpu().tolist()


_embedder: DINOv2MixEmbedder | None = None


def init_embedder() -> None:
    global _embedder
    if _embedder is not None:
        return
    logger.info(f"Загрузка embedding-модели: {config.DINOV2_MODEL}")
    _embedder = DINOv2MixEmbedder()
    logger.success(f"Embedding-модель загружена на {_embedder.device}")


def get_embedder() -> DINOv2MixEmbedder:
    if _embedder is None:
        raise RuntimeError("Embedder не инициализирован")
    return _embedder
