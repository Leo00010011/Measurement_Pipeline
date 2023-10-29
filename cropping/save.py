from tracker.tracker import BoundingBox
from cropping.visualization import crop_image
from cropping.unified_bbox import get_unified_bbox
import cv2
import os

def save_cropped(color, depth, bbox: BoundingBox, data_path, name):
    color = crop_image(color, bbox)
    depth = crop_image(depth, bbox)
    color_path = os.path.join(data_path, 'color', f'{name}.jpg')
    depth_path = os.path.join(data_path, 'depth', f'{name}.png')
    cv2.imwrite(color_path, color)
    cv2.imwrite(depth_path, depth)


def save_all_data_cropped(data):
    unified_bboxes = get_unified_bbox(data)
    bboxes_list = data['bbox_per_frame']
    for i in range(len(bboxes_list)):
        color = cv2.imread(bboxes_list[i]['color_file'])
        depth = cv2.imread(bboxes_list[i]['depth_file'])
        save_cropped(
            color, depth, unified_bboxes[i], 'output', f'sample_{i + 1}')
