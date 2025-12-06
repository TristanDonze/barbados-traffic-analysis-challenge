import cv2
import numpy as np

points = []


def draw_polygon(event, x, y, flags, param):
    global points
    img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 0, 255), 2)
        cv2.imshow("Road Mask Creator", img)


image_path = "./extracted_images/normanniles4_first_frame.jpg"
output_path = "./road_mask/normanniles4_entry_mask.png"
# output_path = "./road_mask/normanniles4_roundpoint_mask.png"
# output_path = "./road_mask/normanniles4_exit_mask.png"

image = cv2.imread(image_path)
original_image = image.copy()

cv2.namedWindow("Road Mask Creator", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Road Mask Creator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Road Mask Creator", image)
cv2.setMouseCallback("Road Mask Creator", draw_polygon, image)

print("Click to define polygon points. Press 'q' to save and exit. Press 'r' to reset.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if len(points) > 2:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            cv2.imwrite(output_path, mask)
            print("Mask saved to", output_path)
        break

    elif key == ord("r"):
        points = []
        image = original_image.copy()
        cv2.imshow("Road Mask Creator", image)
        cv2.setMouseCallback("Road Mask Creator", draw_polygon, image)

cv2.destroyAllWindows()
