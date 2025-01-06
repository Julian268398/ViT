import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms

"""
Kod implementujący konstrukcję transformera
"""


class PatchEmbedding(nn.Module):
    """
    Podział obrazu na patche/łatki

    Parametery:

    img_size: int
        rozmiar zdjęcia (zakładając, że jest kwadratem)

    patch_size: int
        rozmiar patcha/łatki

    input_channels: int
        liczba kanałów wejściowych

    embed_dim: int
        wymiar osadzenia

    __________________________

    Atrybuty:

    n_patches: int
        liczba patchy w zdjęciu

    proj: nn.Conv2d
         Warstwa konwolucyjna, która dokonuje zarówno podziału na patche, jak i ich osadzania.
    """

    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"Image size ({img_size}) must be divisible by patch size ({patch_size}).")

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        """
        Propagacja w przód

        :param x: torch.Tensor
            kształt '(n_samples, input_channels, img_size, img_size)'.

        :return: torch.Tensor
            kształt '(n_samples, n_patches, embed_dim)'
        """
        x = self.proj(x)
        # (n_sample, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)
        # (n_sample, embed_dim, n_patches)
        x = x.transpose(1, 2)
        # (n_sample, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """
    Mechanizm atencji

    Parametery:
        dim: int
            Wymiar wejściowy i wyjściowy cech dla każdego tokena

        n_heads: int
            Liczba głów mechanizmu atencji

        qkv_bias: bool
            Jeśli True, to dodajemy bias (przesunięcie) do projekcji zapytań (query), kluczy (key) i wartości (value)

        attn_p: float
            Prawdopodobieństwo dropout dla tensorów zapytań (query), kluczy (key) i wartości (value)

        proj_p: float
            Prawdopodobieństwo dropout dla tensora wynikowego (po projekcji zapytań)

    _______________
    Atrybuty:
        scale: float
            Stała normalizująca dla iloczynu skalarnego

        qkv: nn.Linear
            Projekcja liniowa dla zapytań (query), kluczy (key) i wartości (value)e

        proj: nn.Linear
            Przekształcenie liniowe, które przyjmuje połączone wyjście ze wszystkich głów atencji i rzutuje je na nową przestrzeń

        attn_drop, proj_drop: nn.Dropout
            Warstwy dropout
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # comes from "All you need is atention"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Propagacja w przód

        :param x: torch.Tensor
            kształt '(n_samples, n_patches + 1, dim)'
        :return: torch.Tensor
            kształt '(n_samples, n_patches + 1, dim)'
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        # (n_samples, n_heads, n_patches + 1, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale
        # (n_samples, n_heads, n_patches +1, n_patches + 1)
        attn = dp.softmax(dim=-1)
        # (n_samples, n_heads, n_patches + 1, n_patches +1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)
        # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)
        # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)
        # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """
    Perceptron wielowarstwowy (ang. Multi-layer Perceptron)

    Parametry:
        in_features: int
            Liczba cech wejściowych

        hidden_features: int
            Liczba neuronów w warstwie ukrytej

        out_features: int
            Liczba cech wyjściowych

        p: float
            Prawdopodobieństwo dropout

    _______________
    Atrybuty:
        fc: nn.Linear
            Pierwsza warstwa liniowa

        act: nn.GELU
            Funkcja aktywacji GELU

        fc2: nn.Linear
            Druga warstwa liniowa

        drop: nn.Dropout
            Warstwa dropout
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Propagacja w przód

        :param x: torch.Tensor
            kształt (n_samples, n_patches + 1, in_features)
        :return: torch.Tensor
            kształt (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(x)
        # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)
        # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)
        # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)
        # (n_samples, n_patches + 1, out_features)

        return x


class Block(nn.Module):
    """
        Blok Transformera

        Parametry:
            dim: int
                Wymiar wejściowy i wyjściowy cech dla każdego tokena

            n_heads: int
                Liczba głów mechanizmu atencji

            mlp_ratio: float
                Określa rozmiar warstwy ukrytej modułu "MLP" względem "dim"

            qkv_bias: bool
                Jeśli True, to dodajemy bias (przesunięcie) do projekcji zapytań (query), kluczy (key) i wartości (value)

            p, attn_p: float
                Prawdopodobieństwo dropout

    _______________
    Atrybuty:
        norm1, norm2: LayerNorm
            Normalizacja warstw

        attn: Attention
            Moduł atencji

        mlp: MLP
            Moduł MLP
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        """
        Propagacja w przód

            :param x: torch.Tensor
                kształt '(n_samples, n_patches + 1, dim)'
            :return: torch.Tensor
                kształt '(n_samples, n_patches + 1, dim)'
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Implementacja transformera wizyjnego

    Parametery:
            img_size: int
                wysokość i szerokość zdjęcia (zakłada się, że jest kwadratowe)

            patch_size: int
                wysokość i szerokość łatki (jest kwadratowa)

            in_chans: int
                liczba kanałów wejściowych

            n_classes: int
                liczba klas, czyli liczba możliwych kategorii, do których model będzie przyporządkowywał dane podczas klasyfikacji.

            embed_dim: int
                Wymiar osadzeń tokenów/patchy

            depth: int
                Liczba bloków

            n_heads: int
                Liczba głów mechanizmu atencji

            mlp_ratio: float
                Określa wymiar ukrytej warstwy modułu 'MLP'

            qkv_bias: bool
                Jeśli True, to dodajemy bias (przesunięcie) do projekcji zapytań (query), kluczy (key) i wartości (value)

            p, attn_p: float
                Prawdopodobieństwo dropout

    _______________
    Atrybuty:
            patch_embed: PatchEmbed
                Instancja warstwy 'PatchEmbed'

            cls_token: nn.Parameter
                parametr, którego można się nauczyć, będzie reprezentował pierwszy token (token klasyfikacyjny)

            pos_embed: nn.Parameter
                parametr, którego można się nauczyć, wektor osadzenia pozycji dla każdego tokena

            transformer_blocks: nn.ModuleList
                Lista bloków transformera

            norm: nn.LayerNorm
                Normalizacja warstwy

            head: nn.Linear
                Głowa klasyfikatora
    """

    def __init__(self, img_size=512, patch_size=16, in_chans=1, n_classes=2, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.p = p
        self.attn_p = attn_p

        # Initialize patch embedding
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                          embed_dim=embed_dim)

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in
             range(depth)]
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classifier head
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Propagacja w przód

        :param x: torch.Tensor
            kształt '(n_samples, in_chans, img_size, img_size)'

        :return: torch.Tensor
            kształt '(n_samples, n_classes)'
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Layer normalization and classification head
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x


#Poniższa częśc wykorzystana była podczas realizacji projektu obsługującego pliki rozszerzenia dcm
class DICOMDataset(Dataset):
    """
    Dataset dla obrazów DICOM.

    Parametery:
    -----------
    dicom_dir: str
        Ścieżka do katalogu zawierającego obrazy DICOM, zorganizowane w podfolderach oznaczających klasy.

    transform: callable
        Transformacje do zastosowania na obrazach (domyślnie brak transformacji).

    __________________________

    Atrybuty:
    ---------
    dicom_dir: str
        Ścieżka do katalogu DICOM.

    transform: callable
        Transformacje stosowane do obrazów.

    image_paths: list[str]
        Lista ścieżek do obrazów DICOM w zestawie danych.

    labels: list[int]
        Lista etykiet przypisanych do obrazów, gdzie:
        - 0 oznacza "benign" (łagodny)
        - 1 oznacza "malignant" (złośliwy)
    """

    def __init__(self, dicom_dir, transform=None):
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Zbieranie ścieżek do plików DICOM i etykiet
        for label in os.listdir(dicom_dir):
            class_folder = os.path.join(dicom_dir, label)
            if os.path.isdir(class_folder):
                for dicom_file in os.listdir(class_folder):
                    if dicom_file.endswith(".dcm"):
                        self.image_paths.append(os.path.join(class_folder, dicom_file))
                        self.labels.append(0 if label == 'benign' else 1)  # 'benign' -> 0, 'malignant' -> 1

    def __len__(self):
        """
            Zwraca długość zbioru danych.

            :return: int
                Liczba obrazów w zbiorze danych.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
            Pobiera element zbioru danych.

            :param idx: int
                Indeks elementu do pobrania.

            :return: tuple
                Obraz przetworzony (torch.Tensor) i jego etykieta (int).
        """
        dicom_path = self.image_paths[idx]
        label = self.labels[idx]

        # Wczytanie obrazu DICOM
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        # Normalizacja obrazu do zakresu [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Konwersja obrazu na format PIL, aby móc stosować transformacje
        image = Image.fromarray((image * 255).astype(np.uint8))  # Zmiana na typ uint8 dla PIL

        if self.transform:
            image = self.transform(image)

        return image, label


# Przygotowanie transformacji
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Konwersja na obraz 1-kanałowy
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalizacja dla obrazów 1-kanałowych
])


# Przygotowanie zbioru treningowego i walidacyjnego
def get_dataloaders(data_dir, batch_size):
    """
        Przygotowanie DataLoaderów dla zbiorów treningowego i walidacyjnego.

        Parametery:
        data_dir: str
            Ścieżka do katalogu z danymi, zawierającego podkatalogi "training" i "validating".

        batch_size: int
            Liczba próbek w jednej partii danych (batchu).


        Zwraca:
        tuple[DataLoader, DataLoader]
            DataLoader dla zbioru treningowego i walidacyjnego.
    """
    train_dataset = DICOMDataset(dicom_dir=f"{data_dir}/training", transform=transform)
    val_dataset = DICOMDataset(dicom_dir=f"{data_dir}/validating", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader