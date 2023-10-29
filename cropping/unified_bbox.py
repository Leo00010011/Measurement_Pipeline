from tracker.tracker import BoundingBox

def get_distorsion(s_bbox: BoundingBox, t_bbox: BoundingBox):
    res_x1 = min(s_bbox.x, t_bbox.x)
    res_y1 = min(s_bbox.y, t_bbox.y)

    res_x2 = max(s_bbox.x + s_bbox.w, t_bbox.x + t_bbox.w)
    res_y2 = max(s_bbox.y + s_bbox.h, t_bbox.y + t_bbox.h)

    res_w = res_x2 - res_x1
    res_h = res_y2 - res_y1

    return BoundingBox(res_x1 - s_bbox.x, res_y1 - s_bbox.y, res_w - s_bbox.w, res_h - s_bbox.h)


def apply_dist(bbox: BoundingBox, dist: BoundingBox):
    bbox.x += dist.x
    bbox.y += dist.y
    bbox.w += dist.w
    bbox.h += dist.h


def copy_bbox(bbox: BoundingBox):
    return BoundingBox(bbox.x, bbox.y, bbox.w, bbox.h)


def apply_dist_to_copy(bbox: BoundingBox, dist: BoundingBox):
    # copy
    bbox_copy = copy_bbox(bbox)
    apply_dist(bbox_copy, dist)
    return bbox_copy


def get_uniform_shape_track_bbox(bbox_list, step_for_rec):
    '''
    Hacer que todoso los bounding box de tracking en un segmento tengan el mismo shape.\n
    Se divide todo el conjunto de bounding boxes en segmentos entre frames que se usó reconocimiento
    y se le asigna que máximo width y height de un segmento a todos los bounding box de tracking.
    '''
    bbox_track = []
    max_h = 0
    max_w = 0
    for i in range(1, len(bbox_list)):
        bbox = BoundingBox(**bbox_list[i]['track_bbox'])
        if i % step_for_rec == 0:
            bbox = BoundingBox(**bbox_list[i]['rect_bbox'])
        if bbox.w > max_w:
            max_w = bbox.w

        if bbox.h > max_h:
            max_h = bbox.h

        if i % step_for_rec == 0:
            if i // step_for_rec == 1:
                bbox_track.append(BoundingBox(
                    bbox_list[0]['rect_bbox']['x'], bbox_list[0]['rect_bbox']['y'], max_w, max_h))
            for j in range(i + 1 - step_for_rec, i + 1):
                bbox_track.append(BoundingBox(
                    bbox_list[j]['track_bbox']['x'], bbox_list[j]['track_bbox']['y'], max_w, max_h))
            max_h = 0
            max_w = 0
        elif i == (len(bbox_list) - 1):
            for j in range((len(bbox_list)//step_for_rec)*step_for_rec + 1, i + 1):
                bbox_track.append(BoundingBox(
                    bbox_list[j]['track_bbox']['x'], bbox_list[j]['track_bbox']['y'], max_w, max_h))
    return bbox_track


def comput_bbox_dist(bbox_list, bbox_track, step_for_rec):
    '''
    Calcular por cada segmento la distorsión necesaria para hacer que el tracking de un segmento contenga el bounding
    box reconocido en el momento que llega a él.
    '''
    bboxes_dist = [BoundingBox(0, 0, 0, 0)]
    for i in range(step_for_rec, len(bbox_list), step_for_rec):
        rect_bbox = BoundingBox(**bbox_list[i]['rect_bbox'])
        if i != (len(bbox_list) - 1):
            rect_bbox = BoundingBox(
                bbox_list[i]['rect_bbox']['x'], bbox_list[i]['rect_bbox']['y'], bbox_track[i + 1].w, bbox_track[i + 1].h)
        track_bbox = bbox_track[i]
        track_bbox = apply_dist_to_copy(track_bbox, bboxes_dist[-1])
        rt_dist = get_distorsion(rect_bbox, track_bbox)
        bboxes_dist.append(rt_dist)
        tr_dist = get_distorsion(track_bbox, rect_bbox)
        for j in range(len(bboxes_dist) - 1):
            apply_dist(bboxes_dist[j], tr_dist)
    return bboxes_dist


def apply_all_distorsions(bbox_list, bbox_track, bboxes_dist, step_for_rec):
    bboxes_result = []
    current_dist_index = 0
    for i in range(len(bbox_list)):
        track_bbox = bbox_track[i]
        apply_dist(track_bbox, bboxes_dist[current_dist_index])
        bboxes_result.append(track_bbox)
        if i != 0 and i % step_for_rec == 0:
            current_dist_index += 1
    return bboxes_result


def get_unified_bbox(data):
    step_for_rec = data['steps_for_rec']
    bbox_list = data['bbox_per_frame']
    # reshape track bboxes:
    bbox_track = get_uniform_shape_track_bbox(bbox_list, step_for_rec)
    # compute distorsions
    bboxes_dist = comput_bbox_dist(bbox_list, bbox_track, step_for_rec)
    # apply all distorsions
    bboxes_result = apply_all_distorsions(
        bbox_list, bbox_track, bboxes_dist, step_for_rec)
    return bboxes_result