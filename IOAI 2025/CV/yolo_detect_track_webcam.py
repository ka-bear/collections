# this is yolo detection + scuffed tracking on webcam 
# it's .py because i can't figure out using webcam on .ipynb 


import os
import pathlib 
from collections import defaultdict 
from typing import Dict, Union, List, Optional, Tuple, Generator, Any 
import math
from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment
import PIL

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.transforms import v2 
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from ultralytics.utils.plotting import Annotator, colors


from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import datetime 


# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


import logging 

# for some reason, for me, i try importing first time it errors and then second time it works so i just do this 
try: 
    import timm
except: 
    pass 

# register new models
from timm.layers import set_layer_config
from timm.models._factory import parse_model_name
from timm.models._helpers import load_state_dict, remap_state_dict
from timm.models._hub import load_model_config_from_hf
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import is_model, model_entrypoint, split_model_name_tag
from timm.data import resolve_data_config, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.volo import VOLO
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.layers.helpers import make_divisible
from timm.layers.mlp import Mlp
from timm.layers.trace_utils import _assert


# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/model/yolo_detector.py

class Detector:
    def __init__(
        self,
        weightss: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ):
        self.face = YOLO(weightss[0]) 
        self.face.fuse() 
        self.person = YOLO(weightss[1]) 
        self.person.fuse() 

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.face.model = self.face.model.half()
            self.person.model = self.person.model.half() 

        self.face_detector_names: Dict[int, str] = self.face.model.names
        self.person_detector_names: Dict[int, str] = self.person.model.names


        # init yolo.predictor
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}
        # self.yolo.predict(**self.detector_kwargs)

    def predict(self, image: Union[np.ndarray, str, "PIL.Image"]):
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return results 

    def track(self, image: Union[np.ndarray, str, "PIL.Image"]):
        face_results: Results = self.face.track(image, persist=True, **self.detector_kwargs)[0]
        person_results: Results = self.person.track(image, persist=True, **self.detector_kwargs)[0]
        return face_results, person_results





# my own stuff 

def pts_to_mc(pt1, pt2): 
    dy = pt2[1] - pt1[1] 
    dx = pt2[0] - pt1[0]
    m = dy/dx 
    c = pt1[1] - m*pt1[0] # c = y - mx 
    return m, c 

class EntranceLine: 
    def __init__(self, a, b): 
        if isinstance(a, (float, int)): # m and c 
            self.m = a 
            self.c = b 
        else: # pt1 and pt2 
            m, c = pts_to_mc(a, b) 
            self.m = m 
            self.c = c 
    
    def above(self, x, y): # above means y < line_y 
        line_y = self.m*x + self.c 
        return (y < line_y) 

    def below(self, x, y): # below means y > line_y 
        line_y = self.m*x + self.c 
        return (y > line_y) 

    def entered(self, entrance_condition, x, y): 
        if entrance_condition == EntranceCondition.BELOW: 
            return self.below(x, y) 
        elif entrance_condition == EntranceCondition.ABOVE: 
            return self.above(x, y) 
        raise ValueError("entrance_condition must be either EntranceCondition.BELOW or EntranceCondition.ABOVE") 
        

''' # attempt at making EntranceCondition.ABOVE's type EntranceCondition: 
class EntranceCondition: 
    def __init__(self, above): 
        self.above = above 

    BELOW = exec('EntranceCondition(0)') 
    ABOVE = exec('EntranceCondition(1)') 
'''

class EntranceCondition: 
    BELOW = 0 
    ABOVE = 1 








# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/structures.py and https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/data/misc.py 

def aggregate_votes_winsorized(ages, max_age_dist=6):
    # Replace any annotation that is more than a max_age_dist away from the median
    # with the median + max_age_dist if higher or max_age_dist - max_age_dist if below
    median = np.median(ages)
    ages = np.clip(ages, median - max_age_dist, median + max_age_dist)
    return np.mean(ages)

def assign_faces(
    persons_bboxes: List[torch.tensor], faces_bboxes: List[torch.tensor], iou_thresh: float = 0.0001
) -> Tuple[List[Optional[int]], List[int]]:
    """
    Assign person to each face if it is possible.
    Return:
        - assigned_faces List[Optional[int]]: mapping of face_ind to person_ind
                                            ( assigned_faces[face_ind] = person_ind ). person_ind can be None
        - unassigned_persons_inds List[int]: persons indexes without any assigned face
    """

    assigned_faces: List[Optional[int]] = [None for _ in range(len(faces_bboxes))]
    unassigned_persons_inds: List[int] = [p_ind for p_ind in range(len(persons_bboxes))]

    if len(persons_bboxes) == 0 or len(faces_bboxes) == 0:
        return assigned_faces, unassigned_persons_inds

    cost_matrix = box_iou(torch.stack(persons_bboxes), torch.stack(faces_bboxes), over_second=True).cpu().numpy()
    persons_indexes, face_indexes = [], []

    if len(cost_matrix) > 0:
        persons_indexes, face_indexes = linear_sum_assignment(cost_matrix, maximize=True)

    matched_persons = set()
    for person_idx, face_idx in zip(persons_indexes, face_indexes):
        ciou = cost_matrix[person_idx][face_idx]
        if ciou > iou_thresh:
            if person_idx in matched_persons:
                # Person can not be assigned twice, in reality this should not happen
                continue
            assigned_faces[face_idx] = person_idx
            matched_persons.add(person_idx)

    unassigned_persons_inds = [p_ind for p_ind in range(len(persons_bboxes)) if p_ind not in matched_persons]

    return assigned_faces, unassigned_persons_inds



def box_iou(box1, box2, over_second=False):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    If over_second == True, return mean(intersection-over-union, (inter / area2))

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    if over_second:
        return (inter / area2 + iou) / 2  # mean(inter / area2, iou)
    else:
        return iou



AGE_GENDER_TYPE = Tuple[float, str]


class PersonAndFaceCrops:
    def __init__(self):
        # int: index of person along results
        self.crops_persons: Dict[int, np.ndarray] = {}

        # int: index of face along results
        self.crops_faces: Dict[int, np.ndarray] = {}

        # int: index of face along results
        self.crops_faces_wo_body: Dict[int, np.ndarray] = {}

        # int: index of person along results
        self.crops_persons_wo_face: Dict[int, np.ndarray] = {}

    def _add_to_output(
        self, crops: Dict[int, np.ndarray], out_crops: List[np.ndarray], out_crop_inds: List[Optional[int]]
    ):
        inds_to_add = list(crops.keys())
        crops_to_add = list(crops.values())
        out_crops.extend(crops_to_add)
        out_crop_inds.extend(inds_to_add)

    def _get_all_faces(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                faces: faces_with_bodies + faces_without_bodies + [None] * len(crops_persons_wo_face)
            if use_persons and not use_faces
                faces: [None] * n_persons
            if not use_persons and use_faces:
                faces: faces_with_bodies + faces_without_bodies
        """

        def add_none_to_output(faces_inds, faces_crops, num):
            faces_inds.extend([None for _ in range(num)])
            faces_crops.extend([None for _ in range(num)])

        faces_inds: List[Optional[int]] = []
        faces_crops: List[Optional[np.ndarray]] = []

        if not use_faces:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons) + len(self.crops_persons_wo_face))
            return faces_inds, faces_crops

        self._add_to_output(self.crops_faces, faces_crops, faces_inds)
        self._add_to_output(self.crops_faces_wo_body, faces_crops, faces_inds)

        if use_persons:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons_wo_face))

        return faces_inds, faces_crops

    def _get_all_bodies(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                persons: bodies_with_faces + [None] * len(faces_without_bodies) + bodies_without_faces
            if use_persons and not use_faces
                persons: bodies_with_faces + bodies_without_faces
            if not use_persons and use_faces
                persons: [None] * n_faces
        """

        def add_none_to_output(bodies_inds, bodies_crops, num):
            bodies_inds.extend([None for _ in range(num)])
            bodies_crops.extend([None for _ in range(num)])

        bodies_inds: List[Optional[int]] = []
        bodies_crops: List[Optional[np.ndarray]] = []

        if not use_persons:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces) + len(self.crops_faces_wo_body))
            return bodies_inds, bodies_crops

        self._add_to_output(self.crops_persons, bodies_crops, bodies_inds)
        if use_faces:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces_wo_body))

        self._add_to_output(self.crops_persons_wo_face, bodies_crops, bodies_inds)

        return bodies_inds, bodies_crops

    def get_faces_with_bodies(self, use_persons: bool, use_faces: bool):
        """
        Return
            faces: faces_with_bodies, faces_without_bodies, [None] * len(crops_persons_wo_face)
            persons: bodies_with_faces, [None] * len(faces_without_bodies), bodies_without_faces
        """

        bodies_inds, bodies_crops = self._get_all_bodies(use_persons, use_faces)
        faces_inds, faces_crops = self._get_all_faces(use_persons, use_faces)

        return (bodies_inds, bodies_crops), (faces_inds, faces_crops)

    def save(self, out_dir="output"):
        ind = 0
        os.makedirs(out_dir, exist_ok=True)
        for crops in [self.crops_persons, self.crops_faces, self.crops_faces_wo_body, self.crops_persons_wo_face]:
            for crop in crops.values():
                if crop is None:
                    continue
                out_name = os.path.join(out_dir, f"{ind}_crop.jpg")
                cv2.imwrite(out_name, crop)
                ind += 1


class PersonAndFaceResult:
    def __init__(self, results: Results):

        self.yolo_results = results
        names = set(results.names.values())
        assert "person" in names and "face" in names

        # initially no faces and persons are associated to each other
        self.face_to_person_map: Dict[int, Optional[int]] = {ind: None for ind in self.get_bboxes_inds("face")}
        self.unassigned_persons_inds: List[int] = self.get_bboxes_inds("person")
        n_objects = len(self.yolo_results.boxes)
        self.ages: List[Optional[float]] = [None for _ in range(n_objects)]
        self.genders: List[Optional[str]] = [None for _ in range(n_objects)]
        self.gender_scores: List[Optional[float]] = [None for _ in range(n_objects)]

    @property
    def n_objects(self) -> int:
        return len(self.yolo_results.boxes)

    @property
    def n_faces(self) -> int:
        return len(self.get_bboxes_inds("face"))

    @property
    def n_persons(self) -> int:
        return len(self.get_bboxes_inds("person"))

    def get_bboxes_inds(self, category: str) -> List[int]:
        bboxes: List[int] = []
        for ind, det in enumerate(self.yolo_results.boxes):
            name = self.yolo_results.names[int(det.cls)]
            if name == category:
                bboxes.append(ind)

        return bboxes

    def get_distance_to_center(self, bbox_ind: int) -> float:
        """
        Calculate euclidian distance between bbox center and image center.
        """
        im_h, im_w = self.yolo_results[bbox_ind].orig_shape
        x1, y1, x2, y2 = self.get_bbox_by_ind(bbox_ind).cpu().numpy()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist([center_x, center_y], [im_w / 2, im_h / 2])
        return dist

    def plot(
        self,
        conf=False,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        labels=True,
        boxes=True,
        probs=True,
        ages=True,
        genders=True,
        gender_probs=False,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.
        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            probs (bool): Whether to plot classification probability
            ages (bool): Whether to plot the age of bounding boxes.
            genders (bool): Whether to plot the genders of bounding boxes.
            gender_probs (bool): Whether to plot gender classification probability
        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """

        # return self.yolo_results.plot()
        colors_by_ind = {}
        for face_ind, person_ind in self.face_to_person_map.items():
            if person_ind is not None:
                colors_by_ind[face_ind] = face_ind + 2
                colors_by_ind[person_ind] = face_ind + 2
            else:
                colors_by_ind[face_ind] = 0
        for person_ind in self.unassigned_persons_inds:
            colors_by_ind[person_ind] = 1

        names = self.yolo_results.names
        annotator = Annotator(
            deepcopy(self.yolo_results.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil,
            example=names,
        )
        pred_boxes, show_boxes = self.yolo_results.boxes, boxes
        pred_probs, show_probs = self.yolo_results.probs, probs

        if pred_boxes and show_boxes:
            for bb_ind, (d, age, gender, gender_score) in enumerate(
                zip(pred_boxes, self.ages, self.genders, self.gender_scores)
            ):
                c, conf, guid = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if guid is None else f"id:{guid} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                if ages and age is not None:
                    label += f" {age:.1f}"
                if genders and gender is not None:
                    label += f" {'F' if gender == 'female' else 'M'}"
                if gender_probs and gender_score is not None:
                    label += f" ({gender_score:.1f})"
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(colors_by_ind[bb_ind], True))

        if pred_probs is not None and show_probs:
            text = f"{', '.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)}, "
            annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        return annotator.result()

    def set_tracked_age_gender(self, tracked_objects: Dict[int, List[AGE_GENDER_TYPE]]):
        """
        Update age and gender for objects based on history from tracked_objects.
        Args:
            tracked_objects (dict[int, list[AGE_GENDER_TYPE]]): info about tracked objects by guid
        """

        for face_ind, person_ind in self.face_to_person_map.items():
            pguid = self._get_id_by_ind(person_ind)
            fguid = self._get_id_by_ind(face_ind)

            if fguid == -1 and pguid == -1:
                # YOLO might not assign ids for some objects in some cases:
                # https://github.com/ultralytics/ultralytics/issues/3830
                continue
            age, gender = self._gather_tracking_result(tracked_objects, fguid, pguid)
            if age is None or gender is None:
                continue
            self.set_age(face_ind, age)
            self.set_gender(face_ind, gender, 1.0)
            if pguid != -1:
                self.set_gender(person_ind, gender, 1.0)
                self.set_age(person_ind, age)

        for person_ind in self.unassigned_persons_inds:
            pid = self._get_id_by_ind(person_ind)
            if pid == -1:
                continue
            age, gender = self._gather_tracking_result(tracked_objects, -1, pid)
            if age is None or gender is None:
                continue
            self.set_gender(person_ind, gender, 1.0)
            self.set_age(person_ind, age)

    def _get_id_by_ind(self, ind: Optional[int] = None) -> int:
        if ind is None:
            return -1
        obj_id = self.yolo_results.boxes[ind].id
        if obj_id is None:
            return -1
        return obj_id.item()

    def get_bbox_by_ind(self, ind: int, im_h: int = None, im_w: int = None) -> torch.tensor:
        bb = self.yolo_results.boxes[ind].xyxy.squeeze().type(torch.int32)
        if im_h is not None and im_w is not None:
            bb[0] = torch.clamp(bb[0], min=0, max=im_w - 1)
            bb[1] = torch.clamp(bb[1], min=0, max=im_h - 1)
            bb[2] = torch.clamp(bb[2], min=0, max=im_w - 1)
            bb[3] = torch.clamp(bb[3], min=0, max=im_h - 1)
        return bb

    def set_age(self, ind: Optional[int], age: float):
        if ind is not None:
            self.ages[ind] = age

    def set_gender(self, ind: Optional[int], gender: str, gender_score: float):
        if ind is not None:
            self.genders[ind] = gender
            self.gender_scores[ind] = gender_score

    @staticmethod
    def _gather_tracking_result(
        tracked_objects: Dict[int, List[AGE_GENDER_TYPE]],
        fguid: int = -1,
        pguid: int = -1,
        minimum_sample_size: int = 10,
    ) -> AGE_GENDER_TYPE:

        assert fguid != -1 or pguid != -1, "Incorrect tracking behaviour"

        face_ages = [r[0] for r in tracked_objects[fguid] if r[0] is not None] if fguid in tracked_objects else []
        face_genders = [r[1] for r in tracked_objects[fguid] if r[1] is not None] if fguid in tracked_objects else []
        person_ages = [r[0] for r in tracked_objects[pguid] if r[0] is not None] if pguid in tracked_objects else []
        person_genders = [r[1] for r in tracked_objects[pguid] if r[1] is not None] if pguid in tracked_objects else []

        if not face_ages and not person_ages:  # both empty
            return None, None

        # You can play here with different aggregation strategies
        # Face ages - predictions based on face or face + person, depends on history of object
        # Person ages - predictions based on person or face + person, depends on history of object

        if len(person_ages + face_ages) >= minimum_sample_size:
            age = aggregate_votes_winsorized(person_ages + face_ages)
        else:
            face_age = np.mean(face_ages) if face_ages else None
            person_age = np.mean(person_ages) if person_ages else None
            if face_age is None:
                face_age = person_age
            if person_age is None:
                person_age = face_age
            age = (face_age + person_age) / 2.0

        genders = face_genders + person_genders
        assert len(genders) > 0
        # take mode of genders
        gender = max(set(genders), key=genders.count)

        return age, gender

    def get_results_for_tracking(self) -> Tuple[Dict[int, Tuple[float, str, List]], Dict[int, Tuple[float, str, List]]]:
        """
        Get objects from current frame
        """
        persons: Dict[int, AGE_GENDER_TYPE, List] = {}
        faces: Dict[int, AGE_GENDER_TYPE, List] = {}

        names = self.yolo_results.names
        pred_boxes = self.yolo_results.boxes
        for _, (det, age, gender, _) in enumerate(zip(pred_boxes, self.ages, self.genders, self.gender_scores)):
            if det.id is None:
                continue
            cat_id, _, guid = int(det.cls), float(det.conf), int(det.id.item())
            name = names[cat_id]
            if name == "person":
                persons[guid] = (age, gender, det.xyxy[0].tolist())
            elif name == "face":
                faces[guid] = (age, gender, det.xyxy[0].tolist())

        return persons, faces

    def associate_faces_with_persons(self):
        face_bboxes_inds: List[int] = self.get_bboxes_inds("face")
        person_bboxes_inds: List[int] = self.get_bboxes_inds("person")

        face_bboxes: List[torch.tensor] = [self.get_bbox_by_ind(ind) for ind in face_bboxes_inds]
        person_bboxes: List[torch.tensor] = [self.get_bbox_by_ind(ind) for ind in person_bboxes_inds]

        self.face_to_person_map = {ind: None for ind in face_bboxes_inds}
        assigned_faces, unassigned_persons_inds = assign_faces(person_bboxes, face_bboxes)

        for face_ind, person_ind in enumerate(assigned_faces):
            face_ind = face_bboxes_inds[face_ind]
            person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None
            self.face_to_person_map[face_ind] = person_ind

        self.unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds]

    def crop_object(
        self, full_image: np.ndarray, ind: int, cut_other_classes: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:

        IOU_THRESH = 0.000001
        MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
        CROP_ROUND_RATE = 0.3
        MIN_PERSON_SIZE = 50

        obj_bbox = self.get_bbox_by_ind(ind, *full_image.shape[:2])
        x1, y1, x2, y2 = obj_bbox
        cur_cat = self.yolo_results.names[int(self.yolo_results.boxes[ind].cls)]
        # get crop of face or person
        obj_image = full_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]

        if cur_cat == "person" and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
            return None

        if not cut_other_classes:
            return obj_image

        # calc iou between obj_bbox and other bboxes
        other_bboxes: List[torch.tensor] = [
            self.get_bbox_by_ind(other_ind, *full_image.shape[:2]) for other_ind in range(len(self.yolo_results.boxes))
        ]

        iou_matrix = box_iou(torch.stack([obj_bbox]), torch.stack(other_bboxes)).cpu().numpy()[0]

        # cut out other objects in case of intersection
        for other_ind, (det, iou) in enumerate(zip(self.yolo_results.boxes, iou_matrix)):
            other_cat = self.yolo_results.names[int(det.cls)]
            if ind == other_ind or iou < IOU_THRESH or other_cat not in cut_other_classes:
                continue
            o_x1, o_y1, o_x2, o_y2 = det.xyxy.squeeze().type(torch.int32)

            # remap current_person_bbox to reference_person_bbox coordinates
            o_x1 = max(o_x1 - x1, 0)
            o_y1 = max(o_y1 - y1, 0)
            o_x2 = min(o_x2 - x1, crop_w)
            o_y2 = min(o_y2 - y1, crop_h)

            if other_cat != "face":
                if (o_y1 / crop_h) < CROP_ROUND_RATE:
                    o_y1 = 0
                if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
                    o_y2 = crop_h
                if (o_x1 / crop_w) < CROP_ROUND_RATE:
                    o_x1 = 0
                if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
                    o_x2 = crop_w

            obj_image[o_y1:o_y2, o_x1:o_x2] = 0

        remain_ratio = np.count_nonzero(obj_image) / (obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2])
        if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
            return None

        return obj_image

    def collect_crops(self, image) -> PersonAndFaceCrops:

        crops_data = PersonAndFaceCrops()
        for face_ind, person_ind in self.face_to_person_map.items():
            face_image = self.crop_object(image, face_ind, cut_other_classes=[])

            if person_ind is None:
                crops_data.crops_faces_wo_body[face_ind] = face_image
                continue

            person_image = self.crop_object(image, person_ind, cut_other_classes=["face", "person"])

            crops_data.crops_faces[face_ind] = face_image
            crops_data.crops_persons[person_ind] = person_image

        for person_ind in self.unassigned_persons_inds:
            person_image = self.crop_object(image, person_ind, cut_other_classes=["face", "person"])
            crops_data.crops_persons_wo_face[person_ind] = person_image

        # uncomment to save preprocessed crops
        # crops_data.save()
        return crops_data







# register mivolo model 

# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/model/cross_bottleneck_attn.py
class CrossBottleneckAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=None,
        qk_ratio=1.0,
        qkv_bias=False,
        scale_pos_embed=False,
    ):
        super().__init__()
        assert feat_size is not None, "A concrete feature size matching expected input (H, W) is required"
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed

        self.qkv_f = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.qkv_p = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)

        # NOTE I'm only supporting relative pos embedding for now
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.norm = nn.LayerNorm([self.dim_out_v * 2, *feat_size])
        mlp_ratio = 4
        self.mlp = Mlp(
            in_features=self.dim_out_v * 2,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv_f.weight, std=self.qkv_f.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.qkv_p.weight, std=self.qkv_p.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def get_qkv(self, x, qvk_conv):
        B, C, H, W = x.shape

        x = qvk_conv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W

        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)

        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)

        return q, k, v

    def apply_attn(self, q, k, v, B, H, W, dropout=None):
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale  # B * num_heads, H * W, H * W
        else:
            attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        if dropout:
            attn = dropout(attn)

        out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        dim = int(C / 2)
        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:, :, :]

        _assert(H == self.pos_embed.height, "")
        _assert(W == self.pos_embed.width, "")

        q_f, k_f, v_f = self.get_qkv(x1, self.qkv_f)
        q_p, k_p, v_p = self.get_qkv(x2, self.qkv_p)

        # person to face
        out_f = self.apply_attn(q_f, k_p, v_p, B, H, W)
        # face to person
        out_p = self.apply_attn(q_p, k_f, v_f, B, H, W)

        x_pf = torch.cat((out_f, out_p), dim=1)  # B, dim_out * 2, H, W
        x_pf = self.norm(x_pf)
        x_pf = self.mlp(x_pf)  # B, dim_out, H, W

        out = self.pool(x_pf)
        return out


# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/model/mivolo_model.py 
"""
Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""



__all__ = ["MiVOLOModel"]  # model_registry will add each entrypoint fn to this


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.96,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": None,
        "classifier": ("head", "aux_head"),
        **kwargs,
    }


default_cfgs = {
    "mivolo_d1_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar", crop_pct=0.96
    ),
    "mivolo_d1_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_384_85.2.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "mivolo_d2_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar", crop_pct=0.96
    ),
    "mivolo_d2_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_384_86.0.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "mivolo_d3_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar", crop_pct=0.96
    ),
    "mivolo_d3_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_448_86.3.pth.tar",
        crop_pct=1.0,
        input_size=(3, 448, 448),
    ),
    "mivolo_d4_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar", crop_pct=0.96
    ),
    "mivolo_d4_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_448_86.79.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "mivolo_d5_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar", crop_pct=0.96
    ),
    "mivolo_d5_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_448_87.0.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "mivolo_d5_512": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_512_87.07.pth.tar",
        crop_pct=1.15,
        input_size=(3, 512, 512),
    ),
}


def get_output_size(input_shape, conv_layer):
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride

    output_size = [
        ((input_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i]) + 1 for i in range(2)
    ]
    return output_size


def get_output_size_module(input_size, stem):
    output_size = input_size

    for module in stem:
        if isinstance(module, nn.Conv2d):
            output_size = [
                (
                    (output_size[i] + 2 * module.padding[i] - module.dilation[i] * (module.kernel_size[i] - 1) - 1)
                    // module.stride[i]
                )
                + 1
                for i in range(2)
            ]

    return output_size


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, img_size=224, stem_conv=False, stem_stride=1, patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans in [3, 6]
        self.with_persons_model = in_chans == 6
        self.use_cross_attn = True

        if stem_conv:
            if not self.with_persons_model:
                self.conv = self.create_stem(stem_stride, in_chans, hidden_dim)
            else:
                self.conv = True  # just to match interface
                # split
                self.conv1 = self.create_stem(stem_stride, 3, hidden_dim)
                self.conv2 = self.create_stem(stem_stride, 3, hidden_dim)
        else:
            self.conv = None

        if self.with_persons_model:

            self.proj1 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )
            self.proj2 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

            stem_out_shape = get_output_size_module((img_size, img_size), self.conv1)
            self.proj_output_size = get_output_size(stem_out_shape, self.proj1)

            self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=self.proj_output_size)

        else:
            self.proj = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim**2

    def create_stem(self, stem_stride, in_chans, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.conv is not None:
            if self.with_persons_model:
                x1 = x[:, :3]
                x2 = x[:, 3:]

                x1 = self.conv1(x1)
                x1 = self.proj1(x1)

                x2 = self.conv2(x2)
                x2 = self.proj2(x2)

                x = torch.cat([x1, x2], dim=1)
                x = self.map(x)
            else:
                x = self.conv(x)
                x = self.proj(x)  # B, C, H, W

        return x


class MiVOLOModel(VOLO):
    """
    Vision Outlooker, the main class of our model
    """

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        pos_drop_rate=0.0, # this part was added, not in the github, as this was a new parameter in volo 
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        use_mix_token=False,
        pooling_scale=2,
    ):
        super().__init__(
            layers,
            img_size,
            in_chans,
            num_classes,
            global_pool,
            patch_size,
            stem_hidden_dim,
            embed_dims,
            num_heads,
            downsamples,
            outlook_attention,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            pos_drop_rate, 
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            post_layers,
            use_aux_head,
            use_mix_token,
            pooling_scale,
        )

        im_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = PatchEmbed(
            img_size=im_size,
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        # step2: tokens learning in the two stages
        x = self.forward_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False, targets=None, epoch=None):
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out

        features = out
        fds_enabled = hasattr(self, "_fds_forward")
        if fds_enabled:
            features = self._fds_forward(features, targets, epoch)

        out = self.head(features)
        if self.aux_head is not None:
            # generate classes in all feature tokens, see token labeling
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]

        return (out, features) if (fds_enabled and self.training) else out

    def forward(self, x, targets=None, epoch=None):
        """simplified forward (without mix token training)"""
        x = self.forward_features(x)
        x = self.forward_head(x, targets=targets, epoch=epoch)
        return x


def _create_mivolo(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")
    #print("VARIANT", variant)
    #print("PRETRAINED", pretrained)
    #print(kwargs)
    return build_model_with_cfg(MiVOLOModel, variant, pretrained, **kwargs)


@register_model
def mivolo_d1_224(pretrained=False, **kwargs):
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_mivolo("mivolo_d1_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d1_384(pretrained=False, **kwargs):
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_mivolo("mivolo_d1_384", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d2_224(pretrained=False, **kwargs):
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d2_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d2_384(pretrained=False, **kwargs):
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d2_384", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d3_224(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d3_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d3_448(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d3_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d4_224(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d4_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d4_448(pretrained=False, **kwargs):
    """VOLO-D4 model, Params: 193M"""
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d4_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_448(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_512(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_512", pretrained=pretrained, **model_args)
    return model






# From https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/model/create_timm_model.py 

"""
Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""



def load_checkpoint(
    model, checkpoint_path, use_ema=True, strict=True, remap=False, filter_keys=None, state_dict_map=None
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    if remap:
        state_dict = remap_state_dict(model, state_dict)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            for filter_key in filter_keys:
                if filter_key in sd_key:
                    if sd_key in state_dict:
                        del state_dict[sd_key]

    rep = []
    if state_dict_map is not None:
        # 'patch_embed.conv1.' : 'patch_embed.conv.'
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    target_name = state_k.replace(target_v, target_k)
                    state_dict[target_name] = state_dict[state_k]
                    rep.append(state_k)
        for r in rep:
            if r in state_dict:
                del state_dict[r]

    incompatible_keys = model.load_state_dict(state_dict, strict=strict if filter_keys is None else False)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: str = "",
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    filter_keys=None,
    state_dict_map=None,
    **kwargs,
):
    """Create a model
    Lookup model's entrypoint function and pass relevant args to create a new model.
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        assert not pretrained_cfg, "pretrained_cfg should not be set when sourcing model from Hugging Face Hub."
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError("Unknown model (%s)" % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, filter_keys=filter_keys, state_dict_map=state_dict_map)

    return model








def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im



def prepare_classification_images(
    img_list: List[Optional[np.ndarray]],
    target_size: int = 224,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    device=None,
) -> torch.tensor:

    prepared_images: List[torch.tensor] = []

    norm_tfm = v2.Normalize(mean=mean, std=std) # this was also added to replace F.normalize as it was updaed 

    for img in img_list:
        if img is None:
            img = torch.zeros((3, target_size, target_size), dtype=torch.float32)
            img = norm_tfm(img)
            img = img.unsqueeze(0)
            prepared_images.append(img)
            continue
        img = class_letterbox(img, new_shape=(target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        prepared_images.append(img)

    if len(prepared_images) == 0:
        return None

    prepared_input = torch.concat(prepared_images)

    if device:
        prepared_input = prepared_input.to(device)

    return prepared_input







# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/model/mi_volo.py 


_logger = logging.getLogger("MiVOLO")
has_compile = hasattr(torch, "compile")


class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2
        self.input_size = 224

    def load_from_ckpt(self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = True if "patch_embed.conv1.0.weight" in state["state_dict"] else False

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError(
                "You can not disable faces and persons together. "
                "Set --with-persons if you want to run with --disable-faces"
            )
        self.input_size = state["state_dict"]["pos_embed"].shape[1] * 16
        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model


class MiVOLO:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
    ):
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        self.meta: Meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)
        if self.verbose:
            _logger.info(f"Model meta:\n{str(self.meta)}")

        model_name = f"mivolo_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        self.param_count = sum([m.numel() for m in self.model.parameters()])
        _logger.info(f"Model {model_name} created, param count: {self.param_count}")

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )

        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        self.model = self.model.to(self.device)

        if torchcompile:
            assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        self.model.eval()
        if self.half:
            self.model = self.model.half()

    def warmup(self, batch_size: int, steps=10):
        if self.meta.with_persons_model:
            input_size = (6, self.input_size, self.input_size)
        else:
            input_size = self.data_config["input_size"]

        input = torch.randn((batch_size,) + tuple(input_size)).to(self.device)

        for _ in range(steps):
            out = self.inference(input)  # noqa: F841

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def inference(self, model_input: torch.tensor) -> torch.tensor:

        with torch.no_grad():
            if self.half:
                model_input = model_input.half()
            output = self.model(model_input)
        return output

    def predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
        if (
            (detected_bboxes.n_objects == 0)
            or (not self.meta.use_persons and detected_bboxes.n_faces == 0)
            or (self.meta.disable_faces and detected_bboxes.n_persons == 0)
        ):
            # nothing to process
            return

        faces_input, person_input, faces_inds, bodies_inds = self.prepare_crops(image, detected_bboxes)

        if faces_input is None and person_input is None:
            # nothing to process
            return

        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        output = self.inference(model_input)

        # write gender and age results into detected_bboxes
        self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

    def fill_in_results(self, output, detected_bboxes, faces_inds, bodies_inds):
        if self.meta.only_age:
            age_output = output
            gender_probs, gender_indx = None, None
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            gender_probs, gender_indx = gender_output.topk(1)

        assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        # per face
        for index in range(output.shape[0]):
            face_ind = faces_inds[index]
            body_ind = bodies_inds[index]

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)

            detected_bboxes.set_age(face_ind, age)
            detected_bboxes.set_age(body_ind, age)

            _logger.info(f"\tage: {age}")

            if gender_probs is not None:
                gender = "male" if gender_indx[index].item() == 0 else "female"
                gender_score = gender_probs[index].item()

                _logger.info(f"\tgender: {gender} [{int(gender_score * 100)}%]")

                detected_bboxes.set_gender(face_ind, gender, gender_score)
                detected_bboxes.set_gender(body_ind, gender, gender_score)

    def prepare_crops(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):

        if self.meta.use_person_crops and self.meta.use_face_crops:
            detected_bboxes.associate_faces_with_persons()

        crops: PersonAndFaceCrops = detected_bboxes.collect_crops(image)
        (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops.get_faces_with_bodies(
            self.meta.use_person_crops, self.meta.use_face_crops
        )

        if not self.meta.use_face_crops:
            assert all(f is None for f in faces_crops)

        faces_input = prepare_classification_images(
            faces_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        if not self.meta.use_person_crops:
            assert all(p is None for p in bodies_crops)

        person_input = prepare_classification_images(
            bodies_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        _logger.info(
            f"faces_input: {faces_input.shape if faces_input is not None else None}, "
            f"person_input: {person_input.shape if person_input is not None else None}"
        )

        return faces_input, person_input, faces_inds, bodies_inds






# from https://github.com/WildChlamydia/MiVOLO/blob/main/mivolo/predictor.py 


def filter_results_by_face(results): # redundant now since it's handled in ress_to_objs 
    cls = results.boxes.cls 
    is1 = [i for i in range(len(cls)) if cls[i] == 1.0] 
    is0 = [i for i in range(len(cls)) if cls[i] == 0.0] 
    face_boxes = results.boxes[is1].xyxy 
    person_boxes = results.boxes[is0].xyxy 

    # find persons that overlap with boxes 
    matching_person_boxes_idxs = [] 
    match_not_found_for_face_idx = [] 
    for fbidx in range(len(face_boxes)): 
        face_box = face_boxes[fbidx]
        found=False 
        for pbidx in range(len(person_boxes)): 
            person_box = person_boxes[pbidx] 
            if boxes_overlap(face_box, person_box): 
                matching_person_boxes_idxs.append(pbidx)  
                found=True 
                break 
        if (not found): 
            match_not_found_for_face_idx.append(fbidx)

    for notfoundidx in sorted(match_not_found_for_face_idx, reverse=True): 
        is1.pop(notfoundidx) 

    return results[is1 + matching_person_boxes_idxs] 
    
def boxes_overlap(face, person): 
    return (person[0] <= face[0]) and (person[1] <= face[1]) and (person[2] >= face[2]) and (person[3] >= face[3]) 


def ress_to_objs(face_res, person_res): 
    raw_person_data = person_res.boxes.data 
    pd = [] 

    for rpdidx in range(raw_person_data.shape[0]): 
        if person_res.boxes.cls[rpdidx] < 0.1: # definitely 0 
            pd.append(raw_person_data[rpdidx].tolist()) 
    person_data = torch.tensor(pd) 

    face_data = face_res.boxes.data 
    
    #print("NAMES:", person_res.names)
    #print("RAW PERSON DATA:", raw_person_data, person_res.names[0])
    #print("PERSON DATA:", person_data, person_data.shape[0])
    #print("FACE DATA:", face_data, face_data.shape[0])
    #1/0
    # match and assign IDs 

    # find persons that overlap with boxes 
    matching_person_boxes_idxss = [] 
    for fbidx in range(face_data.shape[0]): 
        face = face_data[fbidx]
        for pbidx in range(person_data.shape[0]): 
            person = person_data[pbidx] 
            #print("FACE BBOX:", face[:4]) 
            #print("PERSON BBOX:", person[:4])
            if boxes_overlap(face[:4], person[:4]): 
                matching_person_boxes_idxss.append([fbidx, pbidx])  

    #print("MATCHES:", matching_person_boxes_idxss)

    box_data = [] 
    has6 = False 
    for fbidx, pbidx in matching_person_boxes_idxss: 
        person_data_entry = person_data[pbidx].tolist() 
        #person_data_entry[4] += 1 # so that there's no 0 and 0 issue later 
        #personid = person_data_entry[4] 
        face_data_entry = face_data[fbidx].tolist() 
        #if len(face_data_entry)==6: 
            #face_data_entry.insert(4, -personid) 
        #else: 
        #face_data_entry[4] = -personid 

        # set classes again 
        # i blame the weirdness of this code on the completely nonstandard format of stuff here 
        # just set class (last one) to correct one, and ID (4) is smtg else 
        if len(person_data_entry) == 6: 
            person_data_entry.append(0) 
            person_data_entry[4] = 0 # ID uncertain 
            has6 = True 
        else: 
            person_data_entry[6] = 0 
        
        if len(face_data_entry) == 6: 
            face_data_entry.append(0) 
            face_data_entry[4] = 0 # ID uncertain 
            has6 = True 
        else: 
            face_data_entry[6] = 1 
            face_data_entry[4] = -face_data_entry[4] # negate the ID in case, to avoid stuff 

        # add to box_data 
        box_data.append(person_data_entry) 
        box_data.append(face_data_entry)

    box_data = torch.tensor(box_data).reshape((-1,7)) 
    
    if has6: # need to show that id is uncertain so 
        box_data = box_data[:,[0,1,2,3,6,5]] 
        print("HAS6") 
        print(box_data)

    person_res.boxes = Boxes(box_data, person_res.boxes.orig_shape)
    #print("FINAL BOX DATA:",person_res.boxes.data)
    person_res.names = {0:'person', 1:'face'} 

    return person_res 




curdir = pathlib.Path(__file__).parent.resolve() 

yolonames = ['yolov8n_face.pt', 'yolov8n.pt'] 

class Predictor:
    def __init__(self, detector_weightss=[os.path.join(curdir, 'yolo_models', yoloname) for yoloname in yolonames], device='cpu', 
                 checkpoint=os.path.join(curdir, 'yolo_models', 'model_imdb_cross_person_4.22_99.46.pth.tar'), 
                 with_persons=True, disable_faces=False, verbose: bool = False, entrance_line=EntranceLine( (0,180) , (640, 180) ), 
                 entrance_condition=EntranceCondition.BELOW):
        self.detector = Detector(detector_weightss, device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            checkpoint,
            device,
            half=True,
            use_persons=with_persons,
            disable_faces=disable_faces,
            verbose=verbose,
        )
        self.entrance_line:EntranceLine = entrance_line 
        self.entrance_condition:EntranceCondition = entrance_condition 

    def detect(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        #detected_objects: PersonAndFaceResult = self.detector.predict(image)
        detected_objects = self.detector.track(image)
        detected_objects = filter_results_by_face(detected_objects) # ---------------------------------- APPLICAION OF FILTER 
        detected_objects:PersonAndFaceResult = PersonAndFaceResult(detected_objects) 
        return detected_objects 
    
    def recognize(self, image:np.ndarray, detected_objects:PersonAndFaceResult): 
        self.age_gender_model.predict(image, detected_objects)

        out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_track(self, frame, detected_objects_history: Dict[int, List[List]], entereds:list, return_plotted=False, return_det_cnt=False):

        face_res, person_res = self.detector.track(frame)
        detected_objects = ress_to_objs(face_res, person_res) 
        
        #detected_objects = self.detector.track(frame) 
        #detected_objects = filter_results_by_face(detected_objects) # ---------------------------------- APPLICAION OF FILTER 
        
        #print('\n\n')
        #print("DETECTED OBJECTS: ")
        #print(detected_objects) 
        #print('\n') 
        detected_objects:PersonAndFaceResult = PersonAndFaceResult(detected_objects) 
        #starttime = time.time() 
        self.age_gender_model.predict(frame, detected_objects)
        #endtime = time.time() 
        #print("TIME TAKEN FOR MIVOLO:", endtime-starttime) 

        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE, List] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE, List] = current_frame_objs[1]

        # add tr_persons and tr_faces to history
        for guid, datapos in cur_faces.items():
            d1, d2, pos = datapos 
            data = (d1, d2) 
            #print(datapos)
            # not useful for tracking :)
            if None not in data:
                #print("GOT TO POS", pos)
                if self.entrance_line.entered(self.entrance_condition, (pos[0]+pos[2])//2, (pos[1]+pos[3])//2): 
                    #print("ENTERED ALREADY")
                    if guid not in detected_objects_history.keys(): continue # this means it was detected inside already from the start 
                    # done already yay 
                    # average out ages 
                    male_cnt = 0 
                    for age, gen in detected_objects_history[guid]: 
                        male_cnt += int(gen=='male')
                    
                    final_gen = 'male' if (male_cnt*2 >= len(detected_objects_history[guid])) else 'female' # mode 

                    ages = [] # probably will be using median 
                    for age, gen in detected_objects_history[guid]: 
                        if (gen == final_gen): ages.append(age) 
                    final_age = ages[len(ages)//2] # median 

                    entereds.append([final_age, final_gen, datetime.datetime.now()]) # age, gender, time 
                    detected_objects_history.pop(guid, None)
                else: 
                    detected_objects_history[guid].append(data)

        for guid, datapos in cur_persons.items():
            d1, d2, pos = datapos 
            data = (d1, d2) 
            if guid not in detected_objects_history.keys(): continue # has been removed 
            if None not in data:
                detected_objects_history[guid].append(data)

        detected_objects.set_tracked_age_gender(detected_objects_history)

        #print("CUR PERSONS ITEMS") 
        #print(cur_persons)
        #print("DETECTED OBJECTS HISTORY") 
        #print(detected_objects_history) 
        #print(entereds) 

        if return_det_cnt: 
            if return_plotted: return detected_objects_history, entereds, detected_objects.plot(), len(cur_faces.keys())
            else: return detected_objects_history, entereds, len(cur_faces.keys())
        else: 
            if return_plotted: return detected_objects_history, entereds, detected_objects.plot()
            else: return detected_objects_history, entereds 
    
    



# my own stuff 

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

predictor = Predictor(verbose=True)


# actual running loop 
save_video = False 
debug_savedir = "."
filename_prefix = 'yolodetdebug_'
camera_id = 0 


# define entrance line 
entrance_y_coordinate = 180 
entrance_line_xys = (0, entrance_y_coordinate) , (640, entrance_y_coordinate)  
entrance_line = EntranceLine( *entrance_line_xys ) 
# y = mx+c; this is m and c 
entrance_condition = EntranceCondition.BELOW # higher y than line 






def to_dt_format(dt): 
    return dt.strftime("%Y%m%d%H%M%S") 

if save_video: 
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(os.path.join(debug_savedir, 
                                       filename_prefix+'_video_'+to_dt_format(datetime.datetime.now())+'.avi'), 
                                       fourcc, 1, (640, 480) )






# get ready model 

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

predictor = Predictor(verbose=True)







# run 
detected_objects_history: Dict[int, List[List]] = defaultdict(list)

# load entereds if needed 
if os.path.isfile(os.path.join(filename_prefix+"_entereds_save.txt")): 
    with open(os.path.join(filename_prefix+"_entereds_save.txt"), 'r') as f: 
        saved_prev_ckpt = eval(f.readline()) 
        saved_error_time = eval(f.readline()) 
        entereds = eval(f.readline()) 
    if (datetime.datetime.now() - saved_prev_ckpt).total_seconds() < 900: 
        # still the same period yay, keep entereds 
        pass 
    else: 
        # add save function to save results here !! ---------------------------------------------------------------------------
        # reset 
        entereds = [] 

    os.remove(os.path.join(filename_prefix+"_entereds_save.txt"))

else: 
    entereds = [] 



# find next save checkpoint 
next_ckpt = datetime.datetime.now() 
while (next_ckpt.minute%15) != 0: 
    next_ckpt = next_ckpt + datetime.timedelta(minutes=1) 
next_ckpt = next_ckpt.replace(second=0, microsecond=0) 
# get prev checkpoint 
prev_ckpt = next_ckpt - datetime.timedelta(minutes=15) 

def get_next_ckpt(dt): 
    return dt + datetime.timedelta(minutes=15) 

# start running 
try: 
    vs = cv2.VideoCapture(camera_id) #0
    # vs = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) # for pci yuy2

    while True: 
        
        ret, frame = vs.read()
        if not ret: 
            raise ValueError("Failed to read from webcam!")

        detected_objects_history, entereds, frame, det_cnt = predictor.recognize_track(frame, detected_objects_history, entereds, return_plotted=True, return_det_cnt=True) 
        frame = cv2.line(frame, *entrance_line_xys, (255,0,0)) # draw entrance line on frame 
        frame = cv2.putText(frame, str(datetime.datetime.now()), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) # put time 
        frame = cv2.putText(frame, str(datetime.datetime.now()), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # put time 
        if save_video and det_cnt != 0: 
            out.write(frame)
        
        if (datetime.datetime.now() - next_ckpt).total_seconds() > 0: # past the checkpoint 
            # sadd saving code HERE !!! ----------------------------------------------------------------------------

            # clear some storage 
            entereds = [] 
            # no need to clear detected_objects_history as the key is already removed when putting into entereds 
            prev_ckpt = next_ckpt 
            next_ckpt = get_next_ckpt(next_ckpt) 
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        #print(detected_objects_history) 
        #print(entereds)

    raise ValueError("ENDED VIDEO STREAM")


except ValueError as e: 
    error_time = datetime.datetime.now() 

    print() 
    print("ERROR:") 
    print(e) 
    print("\nENDING SESSION!") 
    if save_video: out.release()
    cv2.destroyAllWindows()

    #save_output(entereds, prev_ckpt, er'ror_time)
    with open(os.path.join(filename_prefix+"_entereds_save.txt"), 'w') as f: 
        f.write(repr(prev_ckpt) + '\n' + repr(error_time) + '\n' + repr(entereds) + '\n') 

    print("ENTEREDS:", entereds)


