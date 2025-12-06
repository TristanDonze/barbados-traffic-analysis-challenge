import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "src/mask/extracted_images/normanniles4_first_frame.jpg"

entry_mask = "src/mask/road_mask/normanniles4_entry_mask.png"
roundpoint_mask = "src/mask/road_mask/normanniles4_roundpoint_mask.png"
exit_mask = "src/mask/road_mask/normanniles4_exit_mask.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

entry = cv2.imread(entry_mask, cv2.IMREAD_GRAYSCALE)
roundpoint = cv2.imread(roundpoint_mask, cv2.IMREAD_GRAYSCALE)
exit_ = cv2.imread(exit_mask, cv2.IMREAD_GRAYSCALE)

entry_overlay = np.zeros_like(image)
entry_overlay[:, :, 2] = entry

roundpoint_overlay = np.zeros_like(image)
roundpoint_overlay[:, :, 1] = roundpoint

exit_overlay = np.zeros_like(image)
exit_overlay[:, :, 0] = exit_

combined = image.copy()
alpha = 0.50  # Opacity

combined = cv2.addWeighted(combined, 1, entry_overlay, alpha, 0)
combined = cv2.addWeighted(combined, 1, roundpoint_overlay, alpha, 0)
combined = cv2.addWeighted(combined, 1, exit_overlay, alpha, 0)

plt.figure(figsize=(10, 10))
plt.imshow(combined)
plt.axis('off')
plt.savefig("mask_visualization_4.png", bbox_inches='tight')
