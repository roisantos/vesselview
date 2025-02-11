import torch
import torch.nn.functional as F

class SoftSkeletonize(torch.nn.Module):
    """
    Calcula una esqueletonización suave (soft skeleton) de una imagen.
    
    Entrada:
      - img: Tensor de entrada con forma (B, C, H, W) para imágenes 2D o (B, C, D, H, W) para 3D,
             con valores en el rango [0, 1].
    
    Salida:
      - Un tensor del mismo tamaño que representa el esqueleto suave de la imagen.
    
    Funcionamiento:
      1. Se define una operación de erosión suave mediante pooling máximo negativo.
      2. Se define la dilatación suave usando max pooling.
      3. Se calcula la apertura suave (soft open) como la dilatación de la erosión.
      4. Se inicializa el esqueleto como la diferencia (ReLU) entre la imagen y su apertura.
      5. Se realizan iteraciones en las que se refina el esqueleto acumulando las diferencias.
    """
    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if img.dim() == 4:  # (B, C, H, W)
            p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            return torch.min(p1, p2)
        elif img.dim() == 5:  # (B, C, D, H, W)
            p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
            p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
            p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise ValueError("El tensor de entrada debe ser 4D o 5D.")

    def soft_dilate(self, img):
        if img.dim() == 4:
            return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        elif img.dim() == 5:
            return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        else:
            raise ValueError("El tensor de entrada debe ser 4D o 5D.")

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)
