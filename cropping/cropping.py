import open3d as o3d
import time
import cv2
import os
from reconstruction.utility.file import get_rgbd_file_list2
import numpy as np
from dfu_detector.detect_frcnn_dfu import Faster_RCNN
from tracker.tracker import Tracker, BoundingBox
import json
from cropping.similarity_measures import overlapping_area_bboxes
from cropping.visualization import put_rectangle
from cropping.unified_bbox import get_unified_bbox

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32


def always_rec(rec_bbox, track_bbox):
    return rec_bbox


def create_ulcer_rec():
    '''
    Crea una función que dada una imagen retorna una lista de tuplas
    (BoundingBox,prob)
    '''
    detector = Faster_RCNN(config_output_filename="dfu_detector\\config.pickle",
                           weight_path='dfu_detector\\model_frcnn.hdf5')
    detector.prepare()

    def ulcer_rec(img):
        boxes = detector.get_roi(img)
        fixed_boxes = []
        for box in boxes:
            [x1, y1, x2, y2, prob] = box
            fixed_boxes.append(
                (BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), float(prob)))
        return fixed_boxes
    return ulcer_rec


def choose_roi_rois(img: np.ndarray, bboxes):
    # putting bboxes
    img_copy = img.copy()
    for index, (bbox, prob) in enumerate(bboxes):
        put_rectangle(img_copy, bbox, f'ulcer_{index}: {int(100*prob)}')
    cv2.imshow('Rois', img_copy)
    # selecting bboxes
    selected_bbox = None
    selected_prob = None
    index = None
    while True:
        index = cv2.waitKey() - 48
        if 0 <= index < len(bboxes):
            selected_bbox, selected_prob = bboxes[index]
            break
    # showing selected bbox
    img_copy = img.copy()
    put_rectangle(img_copy, selected_bbox,
                  f'ulcer_{index}: {int(100*selected_prob)}')
    cv2.imshow('Rois', img_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return selected_bbox


def rectifie(img, rec_bboxes, track_bbox=None):
    final = None
    if len(rec_bboxes) > 1:
        # Si hay más de una ulcera reconocida toca desambiguar
        if track_bbox is None:
            img_copy = img.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            final = choose_roi_rois(img_copy, rec_bboxes)
        else:
            # Calcular el solapamiento del bbox trackeado con cada uno reconocido
            overlapp = [overlapping_area_bboxes(
                rec_bbox, track_bbox) for rec_bbox, _ in rec_bboxes]
            # Obtener el máximo
            max_index = 0
            max_value = overlapp[0]
            for index, area in enumerate(overlapp):
                if area > max_value:
                    max_value = area
                    max_index = index
            final, _ = rec_bboxes[max_index]
    else:
        final, _ = rec_bboxes[0]
    return final


def cropp_the_images_with_stepped_recognition(config, tracker: Tracker, ulcer_rec, steps_for_rec=20, rectifie_func=always_rec):
    print('Loading images')
    [color_files, depth_files] = get_rgbd_file_list2(
        config["path_dataset"], False, config['frame_step'])
    rec_bboxes = None
    current_bbox = None
    track_bbox = None
    bbox_data = {
        'steps_for_rec': steps_for_rec,
        'bbox_per_frame': [],
        'time_per_tracking': [],
        'time_per_rec': [],
    }
    total = len(color_files)
    start_total_time = time.time()
    for index, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        print(f'cropping: {index + 1}/{total}')
        # read the images
        current_img_data = {
            'color_file': color_file,
            'depth_file': depth_file
        }
        color = np.asarray(o3d.io.read_image(color_file))
        # cropp the images
        if index % steps_for_rec == 0:
            start = time.time()  # time for rec
            rec_bboxes = ulcer_rec(color)
            bbox_data['time_per_rec'].append(time.time() - start)
            serializable_rec_bboxes = [(bbox.get_as_dict(), prob)
                                       for bbox, prob in rec_bboxes]
            current_img_data['rec_bbox'] = serializable_rec_bboxes
            if index == 0:
                current_bbox = rectifie_func(color, rec_bboxes, None)
                current_img_data['rect_bbox'] = current_bbox.get_as_dict()
            else:
                start = time.time()
                track_bbox: BoundingBox = tracker.update(color)
                bbox_data['time_per_tracking'].append(time.time() - start)
                current_bbox = rectifie_func(color, rec_bboxes, track_bbox)
                current_img_data['rect_bbox'] = current_bbox.get_as_dict()
                current_img_data['track_bbox'] = track_bbox.get_as_dict()
            tracker.set_bbox_to_follow(color, current_bbox)
        else:
            start = time.time()
            current_bbox = tracker.update(color)
            bbox_data['time_per_tracking'].append(time.time() - start)
            current_img_data['track_bbox'] = current_bbox.get_as_dict()
        # save the images
        bbox_data['bbox_per_frame'].append(current_img_data)
    bbox_data['total_time'] = time.time() - start_total_time

    with open('bboxes.json', 'w') as f:
        json.dump(bbox_data, f)
    return bbox_data


def cropp_the_images_with_initial_bbox_and_tracking(config, tracker: Tracker):
    [color_files, depth_files] = get_rgbd_file_list2(
        config["path_dataset"], False, config['frame_step'])
    track_bbox: BoundingBox = None
    bboxes_data = {'bboxes': []}
    all_time_start = None
    for color_file, depth_file in zip(color_files, depth_files):
        color = np.asarray(o3d.io.read_image(color_file))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        step_time = time.time()
        if not track_bbox:
            while not track_bbox:
                track_bbox = cv2.selectROI(
                    'Select the ulcer', color, fromCenter=False, showCrosshair=False)
            cv2.destroyAllWindows()
            tracker.set_bbox_to_follow(color, track_bbox)
            track_bbox = BoundingBox(*track_bbox).get_as_dict()
            # Empezar a contar el tiempo después de seleccionar el ROI
            all_time_start = time.time()
        else:
            track_bbox = tracker.update(color)
            track_bbox = track_bbox.get_as_dict()

        bboxes_data['bboxes'].append({
            'color_path': color_file,
            'depth_path': depth_file,
            'bbox': track_bbox,
            'time': time.time() - step_time
        })

    bboxes_data['all_time'] = time.time() - all_time_start
    outpath = os.path.join(config['path_dataset','bboxes_data.json'])
    with open(outpath, 'w') as f:
        json.dump(bboxes_data, f)
    return bboxes_data




