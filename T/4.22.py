import numpy as np
from PIL import Image
from torchvision import transforms
from imgaug import augmenter as iaa

class IaaTransforms:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SaltAndPepper(p=(0.03, 0.07)),
            iaa.Rain(speed=(0.3, 0.7))
        ])

    def __call__(self, images):
        images = np.asarray(images)
        augmented = self.seq.augment_image(images)
        return Images.fromarray(augmented)

trans = IaaTransforms.Compose([
    IaaTransforms()
])

print(trans(np.ones((1, 1, 3, 3), dtype=np.uint8)))