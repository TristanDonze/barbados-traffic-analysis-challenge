import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
mask_dir = os.path.join(current_dir, "..", "mask", "road_mask")


def get_mask(filename):
    path = os.path.join(mask_dir, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    return img


mask_entry_1 = get_mask("normanniles1_entry_mask.png")
mask_round_1 = get_mask("normanniles1_roundpoint_mask.png")
mask_exit_1 = get_mask("normanniles1_exit_mask.png")

mask_total_1 = cv2.bitwise_or(mask_entry_1, mask_round_1)
mask_total_1 = cv2.bitwise_or(mask_total_1, mask_exit_1)

mask_entry_2 = get_mask("normanniles2_entry_mask.png")
mask_round_2 = get_mask("normanniles2_roundpoint_mask.png")
mask_exit_2 = get_mask("normanniles2_exit_mask.png")

mask_total_2 = cv2.bitwise_or(mask_entry_2, mask_round_2)
mask_total_2 = cv2.bitwise_or(mask_total_2, mask_exit_2)

mask_entry_3 = get_mask("normanniles3_entry_mask.png")
mask_round_3 = get_mask("normanniles3_roundpoint_mask.png")
mask_exit_3 = get_mask("normanniles3_exit_mask.png")

mask_total_3 = cv2.bitwise_or(mask_entry_3, mask_round_3)
mask_total_3 = cv2.bitwise_or(mask_total_3, mask_exit_3)

mask_entry_4 = get_mask("normanniles4_entry_mask.png")
mask_round_4 = get_mask("normanniles4_roundpoint_mask.png")
mask_exit_4 = get_mask("normanniles4_exit_mask.png")

mask_total_4 = cv2.bitwise_or(mask_entry_4, mask_round_4)
mask_total_4 = cv2.bitwise_or(mask_total_4, mask_exit_4)

all_masks = {
    "normanniles1": {
        0: mask_entry_1,
        1: mask_exit_1,
        2: mask_round_1,
        3: mask_total_1,
    },
    "normanniles2": {
        0: mask_entry_2,
        1: mask_exit_2,
        2: mask_round_2,
        3: mask_total_2,
    },
    "normanniles3": {
        0: mask_entry_3,
        1: mask_exit_3,
        2: mask_round_3,
        3: mask_total_3,
    },
    "normanniles4": {
        0: mask_entry_4,
        1: mask_exit_4,
        2: mask_round_4,
        3: mask_total_4,
    },
}
