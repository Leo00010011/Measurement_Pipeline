import cv2
from tracker.tracker import BoundingBox
from cropping.unified_bbox import get_unified_bbox

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32

def put_rectangle(img, bbox, textLabel, color=(255, 0, 0)):
    x1, y1 = bbox.x, bbox.y
    x2, y2 = x1 + bbox.w, y1 + bbox.h
    cv2.rectangle(img, (x1, y1), (x2, y2), color)
    (retval, baseLine) = cv2.getTextSize(
        textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
    textOrg = (x1, y1-0)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5),
                  (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5),
                  (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    cv2.putText(img, textLabel, textOrg,
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img


def add_bboxes_simple_tracking(index, img, data):
    img = put_rectangle(img, BoundingBox(**data[index]['bbox']), 'ulcer')


unified_bboxes = None

def add_bboxes_tracking_with_rec(index, img, data):
    steps_for_rec = data['steps_for_rec']
    global unified_bboxes
    if unified_bboxes is None:
        print('Computing unified bboxes')
        unified_bboxes = get_unified_bbox(data)
    put_rectangle(img, unified_bboxes[index], 'Unified', (255, 255, 0))

    if index == 0:
        for i, [bbox, _] in enumerate(data['bbox_per_frame'][index]['rec_bbox']):
            put_rectangle(img, BoundingBox(**bbox),
                          f'rec_bbox_{i + 1}', (0, 0, 255))
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['rect_bbox']), 'rect_bbox', (0, 255, 0))
    elif index % steps_for_rec == 0:
        for i, [bbox, _] in enumerate(data['bbox_per_frame'][index]['rec_bbox']):
            put_rectangle(img, BoundingBox(**bbox),
                          f'rec_bbox_{i + 1}', (0, 0, 255))
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['rect_bbox']), 'rect_bbox', (0, 255, 0))
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['track_bbox']), 'track_bbox')
    else:
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['track_bbox']), 'track_bbox')
        
def show_bboxes(bboxes, add_bboxes_func):
    index = 0
    count = 0
    while True:
        color = cv2.imread(bboxes['bbox_per_frame'][index]['color_file'])
        add_bboxes_func(index, color, bboxes)
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(bboxes['bbox_per_frame']):
                index = len(bboxes['bbox_per_frame']) - 1
        elif code == SPACE_BAR:
            cv2.imwrite(
                f'results with only tracker\\sample_{count}.jpg', color)
            count += 1
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()

def crop_image(img, bbox: BoundingBox):
    return img[bbox.y: bbox.y + bbox.h, bbox.x: bbox.x + bbox.w]


def show_bboxes_cropped(bboxes):
    index = 0
    count = 0
    unified_bboxes = get_unified_bbox(bboxes)
    while True:
        color = cv2.imread(bboxes['bbox_per_frame'][index]['color_file'])
        color = crop_image(color,unified_bboxes[index])
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(bboxes['bbox_per_frame']):
                index = len(bboxes['bbox_per_frame']) - 1
        elif code == SPACE_BAR:
            cv2.imwrite(
                f'results with only tracker\\sample_{count}.jpg', color)
            count += 1
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()
        
# acumular las distorciones mínimas
# t1 + p12 + p23 + p34 + p45
# t2 + p21 + p23 + p34 + p45
# t3 + p32       + p34 + p45
# t4 + p43             + p45
# t5 + p54