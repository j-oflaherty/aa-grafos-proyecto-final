# %%
import start

# %%
from vision_gnn.data import MNISTDataModule

datamodule = MNISTDataModule()
datamodule.setup()
# %%
train_data = datamodule.train_dataloader()

# %%
for batch in train_data:
    imgs, labels = batch
    break
# %%
import matplotlib.pyplot as plt

plt.imshow(imgs[7].permute(1, 2, 0).numpy())
# %%
from vision_gnn.lightning_module import VigLightningModule

model = VigLightningModule()
# %%
for b in train_data:
    out = model.forward(b[0])
    break
# %%
print(out.shape)
