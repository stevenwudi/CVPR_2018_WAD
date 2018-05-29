import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


instance_name = '/media/samsumg_1tb/CITYSCAPE/gtFine/val/lindau/lindau_000000_000019_gtFine_instanceIds.png'
label_name = instance_name.replace('instanceIds', 'labelIds')

instance = np.array(Image.open(instance_name))
label = np.array(Image.open(label_name))

plt.close()
plt.figure()
plt.imshow(instance)
plt.figure()
plt.imshow(label)



