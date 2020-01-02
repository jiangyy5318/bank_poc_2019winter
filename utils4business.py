# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from shapely.geometry import box, Polygon
import sys, os
import cv2 as cv


def genetic_iou_cal(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """

    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    area_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C's area
    area_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G's area

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    insect_area = w * h
    print(area_rec1,area_rec2,insect_area)
    res_iou = float(insect_area) / (area_rec1 + area_rec2 - insect_area)
    return res_iou


class iou_calc_class(object):
    def __init__(self, method='rectangle', object_rect=(), polygon_rect=[]):
        """
        :param method: rectangle, polygon
        if rectangle, we get iou calc by means of genetic functions
        if polygon, we will involve intersect of two polygon and its area
        """
        self.method = method
        self.object_rect = None
        self.object_rect_area = 0.0
        if self.method is 'rectangle':
            self.object_rect = object_rect
        elif self.method is 'polygon':
            polygon_rect_np = None
            try:
                polygon_rect_np = np.array(polygon_rect)
            except Exception as e:
                print(e.with_traceback())
                print('polygon rect is not n*2 format')
            self.object_rect = Polygon(polygon_rect_np).convex_hull
            self.object_rect_area = self.object_rect.area
        else:
            raise NotImplemented

    def calc_iou(self, predict_rect=()):
        if self.method is 'rectangle':
            print(self.object_rect, predict_rect)
            return genetic_iou_cal(self.object_rect, predict_rect)
        elif self.method is 'polygon':
            predict_box = box(predict_rect[0], predict_rect[1], predict_rect[2], predict_rect[3])
            predict_box_area = predict_box.area
            # polygon_iou = 0.0
            if not predict_box.intersects(self.object_rect):
                polygon_iou = 0.0
            else:
                intersect_area = predict_box.intersection(self.object_rect).area
                polygon_iou = intersect_area / (predict_box_area + self.object_rect_area - intersect_area)
            return polygon_iou
        else:
            raise NotImplemented


def main_test_polygon_iou():
    box1 = [(2, 0), (2, 2), (0, 0), (0, 2)]
    rect1 = [0, 0, 2, 2]
    box2 = [(1, 1), (4, 1), (4, 4), (1, 4)]
    rect2 = [1, 1, 4, 4]
    genetic_iou_cls = iou_calc_class(method='rectangle', object_rect=rect1)
    print(genetic_iou_cls.calc_iou(predict_rect=rect2))
    poly_iou_cls = iou_calc_class(method='polygon', polygon_rect=box1)
    print(poly_iou_cls.calc_iou(predict_rect=rect2))


def detect_roi_exist_people(detections, roi, threshold):
    iou_list = []
    people_exist = False
    for detection in detections:
        coor = np.array(detection[:4], dtype=np.int32)
        xmin, ymin, xmax, ymax = coor[0], coor[1], coor[2], coor[3]

        iou_list.append(genetic_iou_cal((xmin, ymin, xmax, ymax), roi))
        people_exist = (max(iou_list) >= threshold)

    return people_exist, iou_list


def cv_add_boxes_polygons(raw_image, roi_list=[], roi_name_list=[]):
    assert len(roi_list) == len(roi_name_list) or len(roi_name_list) == 0
    cv_image = raw_image
    for idx in range(len(roi_list)):
        _roi = roi_list[idx]
        _roi_name = roi_name_list[idx] if len(roi_name_list) > 0 else ''
        if len(_roi) == 2:
            cv.rectangle(cv_image, _roi[0], _roi[1], (0, 255, 255), 1)
        else:
            cv.polylines(cv_image, _roi, True)
        cv.putText(cv_image, _roi_name, _roi[0],
                   cv.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
    return cv_image


# def image_capture(video_format='.mp4', tuple_list=[]):
#     for item in tuple_list:
#         video_file = os.path.join(videoPath, item[0] + video_format)
#         assert os.path.exists(video_file)
#         video_cap = cv.VideoCapture(video_file)
#         for i in range(int(item[1])):
#             ret, frame = video_cap.read()
#         save_path = os.path.join(videoPath, item[0] + '-' + str(item[1]) + '.jpg')
#         cv.imwrite(save_path, frame)
#         video_cap.release()


if __name__ == '__main__':
    main_test_polygon_iou()