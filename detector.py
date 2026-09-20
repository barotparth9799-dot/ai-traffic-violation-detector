import cv2
import os
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class TrafficViolationDetector:
    """
    Traffic violation detector.

    Supported violations:
    1. Triple Riding
    2. Riding without Helmet

    The detector also provides:
    - person detection
    - motorcycle detection
    - rider-to-motorcycle association
    - helmet/no-helmet detection
    - evidence image saving
    - motorcycle crop for OCR
    """

    def __init__(self):
        self.project_dir = Path(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.evidence_dir = self.project_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # COCO MODEL
        # ---------------------------------------------------------

        coco_path = self.project_dir / "yolov8n.pt"

        print("Loading YOLOv8 COCO model...")

        try:
            if coco_path.exists():
                self.coco_model = YOLO(str(coco_path))
            else:
                self.coco_model = YOLO("yolov8n.pt")

            print("COCO model loaded successfully.")

        except Exception as e:
            print("ERROR loading COCO model:", e)
            self.coco_model = None

        # ---------------------------------------------------------
        # HELMET MODEL
        # ---------------------------------------------------------

        self.helmet_model = None

        helmet_path = self.project_dir / "weights" / "best.pt"

        print("Loading helmet detection model...")

        try:
            if helmet_path.exists():
                self.helmet_model = YOLO(str(helmet_path))
                print("Local helmet model loaded successfully.")
            else:
                print(
                    "WARNING: weights\\best.pt was not found."
                )
                print(
                    "Helmet detection will be unavailable."
                )

        except Exception as e:
            print("ERROR loading helmet model:", e)
            self.helmet_model = None

    # =========================================================
    # BASIC BOX UTILITIES
    # =========================================================

    @staticmethod
    def box_center(box):
        x1, y1, x2, y2 = box

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

    @staticmethod
    def intersection_over_box(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (
            (x2 - x1) *
            (y2 - y1)
        )

        box_area = max(
            1.0,
            (ax2 - ax1) *
            (ay2 - ay1)
        )

        return float(intersection / box_area)

    def calculate_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection = max(
            0,
            x2 - x1
        ) * max(
            0,
            y2 - y1
        )

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    # =========================================================
    # MOTORCYCLE / RIDER ASSOCIATION
    # =========================================================

    def is_overlapping(
        self,
        person_box,
        motorcycle_box
    ):
        """
        Determines whether a person is plausibly riding
        a motorcycle.

        This is intentionally tolerant because multiple
        riders on one scooter can overlap heavily.
        """

        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = motorcycle_box

        person_width = max(
            1.0,
            px2 - px1
        )

        person_height = max(
            1.0,
            py2 - py1
        )

        # -----------------------------------------------------
        # Direct intersection
        # -----------------------------------------------------

        x1 = max(px1, mx1)
        y1 = max(py1, my1)
        x2 = min(px2, mx2)
        y2 = min(py2, my2)

        if x2 > x1 and y2 > y1:

            intersection = (
                (x2 - x1) *
                (y2 - y1)
            )

            person_area = (
                person_width *
                person_height
            )

            overlap_ratio = (
                intersection /
                person_area
            )

            if overlap_ratio >= 0.10:
                return True

        # -----------------------------------------------------
        # Expanded motorcycle region
        # -----------------------------------------------------

        motorcycle_width = max(
            1.0,
            mx2 - mx1
        )

        motorcycle_height = max(
            1.0,
            my2 - my1
        )

        expanded = [
            mx1 - motorcycle_width * 0.35,
            my1 - motorcycle_height * 0.35,
            mx2 + motorcycle_width * 0.35,
            my2 + motorcycle_height * 0.35
        ]

        ex1, ey1, ex2, ey2 = expanded

        center_x = (
            (px1 + px2) / 2.0
        )

        center_y = (
            (py1 + py2) / 2.0
        )

        motorcycle_center_x = (
            (mx1 + mx2) / 2.0
        )

        motorcycle_center_y = (
            (my1 + my2) / 2.0
        )

        horizontal_distance = abs(
            center_x -
            motorcycle_center_x
        )

        vertical_distance = (
            motorcycle_center_y -
            center_y
        )

        # Person should generally be above the motorcycle.
        if (
            horizontal_distance
            <= motorcycle_width * 1.15
            and
            vertical_distance
            >= -motorcycle_height * 0.45
            and
            center_x >= ex1
            and
            center_x <= ex2
            and
            center_y >= ey1
            and
            center_y <= ey2
        ):
            return True

        return False

    def rider_motorcycle_score(
        self,
        person_box,
        motorcycle_box
    ):
        """
        Calculates how strongly a person appears to be
        associated with a motorcycle.

        Multiple people can be associated with the same
        motorcycle. This is important for triple riding.
        """

        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = motorcycle_box

        person_center_x, person_center_y = (
            self.box_center(person_box)
        )

        motorcycle_center_x, motorcycle_center_y = (
            self.box_center(motorcycle_box)
        )

        direct_overlap = (
            self.intersection_over_box(
                person_box,
                motorcycle_box
            )
        )

        motorcycle_width = max(
            1.0,
            mx2 - mx1
        )

        motorcycle_height = max(
            1.0,
            my2 - my1
        )

        expanded_motorcycle = [
            mx1 - motorcycle_width * 0.35,
            my1 - motorcycle_height * 0.35,
            mx2 + motorcycle_width * 0.35,
            my2 + motorcycle_height * 0.35
        ]

        expanded_overlap = (
            self.intersection_over_box(
                person_box,
                expanded_motorcycle
            )
        )

        horizontal_distance = abs(
            person_center_x -
            motorcycle_center_x
        )

        horizontal_score = max(
            0.0,
            1.0 -
            (
                horizontal_distance /
                motorcycle_width
            )
        )

        vertical_distance = (
            motorcycle_center_y -
            person_center_y
        )

        if vertical_distance >= 0:
            vertical_score = min(
                1.0,
                vertical_distance /
                motorcycle_height
            )
        else:
            vertical_score = 0.0

        score = (
            direct_overlap * 0.50
            +
            expanded_overlap * 0.25
            +
            horizontal_score * 0.20
            +
            vertical_score * 0.05
        )

        # Very far horizontally = unlikely rider.
        if (
            horizontal_distance
            > motorcycle_width * 1.35
        ):
            score *= 0.20

        return float(score)

    def associate_riders_with_motorcycles(
        self,
        people,
        motorcycles
    ):
        """
        Associates people with motorcycles.

        IMPORTANT:
        Multiple people are allowed to belong to the
        SAME motorcycle.

        Each person can still belong to only one motorcycle.
        """

        candidate_pairs = []

        for motorcycle_index, motorcycle in enumerate(
            motorcycles
        ):

            motorcycle_box = motorcycle["box"]

            for person_index, person in enumerate(
                people
            ):

                person_box = person["box"]

                score = self.rider_motorcycle_score(
                    person_box,
                    motorcycle_box
                )

                # Tolerant threshold specifically for
                # crowded scooters / triple riding.
                if score >= 0.22:

                    candidate_pairs.append(
                        (
                            score,
                            motorcycle_index,
                            person_index
                        )
                    )

        # Strongest matches first.
        candidate_pairs.sort(
            key=lambda x: x[0],
            reverse=True
        )

        motorcycle_riders = {}
        assigned_people = set()

        for (
            score,
            motorcycle_index,
            person_index
        ) in candidate_pairs:

            # One person cannot belong to two motorcycles.
            if person_index in assigned_people:
                continue

            if motorcycle_index not in motorcycle_riders:
                motorcycle_riders[
                    motorcycle_index
                ] = []

            motorcycle_riders[
                motorcycle_index
            ].append(
                person_index
            )

            assigned_people.add(
                person_index
            )

        associations = []

        for (
            motorcycle_index,
            riders
        ) in motorcycle_riders.items():

            associations.append(
                (
                    motorcycle_index,
                    len(riders),
                    riders
                )
            )

        return associations

    # =========================================================
    # HELMET DETECTION
    # =========================================================

    def detect_helmet_objects(
        self,
        image,
        confidence=0.08,
        imgsz=1280
    ):
        """
        Detect helmet/no-helmet objects.

        Expected model classes:

        0 = With Helmet
        1 = Without Helmet
        """

        helmets = []
        no_helmets = []

        if self.helmet_model is None:
            return helmets, no_helmets

        try:

            results = self.helmet_model(
                image,
                conf=confidence,
                imgsz=imgsz,
                verbose=False
            )[0]

        except Exception as e:

            print(
                "Helmet detection error:",
                e
            )

            return helmets, no_helmets

        if results.boxes is None:
            return helmets, no_helmets

        for box in results.boxes:

            try:

                cls_id = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                )

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            except Exception:
                continue

            if cls_id == 0:

                helmets.append(
                    {
                        "box": coords,
                        "confidence": conf,
                        "class_name": "With Helmet"
                    }
                )

            elif cls_id == 1:

                no_helmets.append(
                    {
                        "box": coords,
                        "confidence": conf,
                        "class_name": "Without Helmet"
                    }
                )

        return helmets, no_helmets

    def get_rider_head_region(
        self,
        rider_box
    ):
        """
        Returns the upper portion of a rider box,
        where helmet/no-helmet detections normally occur.
        """

        x1, y1, x2, y2 = rider_box

        height = max(
            1,
            y2 - y1
        )

        return [
            x1 - 0.20 * (x2 - x1),
            y1 - 0.10 * height,
            x2 + 0.20 * (x2 - x1),
            y1 + 0.55 * height
        ]

    def match_helmet_to_rider(
        self,
        image,
        rider_box
    ):
        """
        Matches the helmet model's result to a rider.

        Returns the strongest helmet/no-helmet match.
        """

        helmets, no_helmets = (
            self.detect_helmet_objects(
                image
            )
        )

        head_region = (
            self.get_rider_head_region(
                rider_box
            )
        )

        candidates = []

        # -----------------------------------------------------
        # NO HELMET
        # -----------------------------------------------------

        for detection in no_helmets:

            score = (
                self.intersection_over_box(
                    detection["box"],
                    head_region
                )
            )

            if score > 0.05:

                candidates.append(
                    (
                        score,
                        detection
                    )
                )

        # -----------------------------------------------------
        # HELMET
        # -----------------------------------------------------

        for detection in helmets:

            score = (
                self.intersection_over_box(
                    detection["box"],
                    head_region
                )
            )

            if score > 0.05:

                candidates.append(
                    (
                        score,
                        detection
                    )
                )

        if not candidates:
            return None

        # Highest spatial match first.
        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]["confidence"]
            ),
            reverse=True
        )

        return candidates[0][1]

    # =========================================================
    # EVIDENCE
    # =========================================================

    def save_evidence(
        self,
        image,
        violation_type
    ):
        timestamp = int(
            time.time() * 1000
        )

        safe_name = (
            violation_type
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"{safe_name}_{timestamp}.jpg"
        )

        output_path = (
            self.evidence_dir /
            filename
        )

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if not success:
            raise RuntimeError(
                "Could not save evidence image."
            )

        return str(output_path)

    # =========================================================
    # MOTORCYCLE CROP
    # =========================================================

    def get_motorcycle_crop(
        self,
        image,
        motorcycle_box
    ):
        h, w = image.shape[:2]

        x1, y1, x2, y2 = [
            int(v)
            for v in motorcycle_box
        ]

        x1 = max(
            0,
            min(x1, w - 1)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        x2 = max(
            x1 + 1,
            min(x2, w)
        )

        y2 = max(
            y1 + 1,
            min(y2, h)
        )

        return image[
            y1:y2,
            x1:x2
        ].copy()

    # =========================================================
    # MAIN DETECTION PIPELINE
    # =========================================================

    def detect_violations(
        self,
        image_path_or_frame
    ):
        """
        Complete traffic violation detection pipeline.

        Detects:
        - Triple Riding
        - Riding without Helmet
        """

        # -----------------------------------------------------
        # LOAD IMAGE
        # -----------------------------------------------------

        if isinstance(
            image_path_or_frame,
            str
        ):

            image = cv2.imread(
                image_path_or_frame
            )

            if image is None:
                raise ValueError(
                    "Could not load image: "
                    +
                    str(image_path_or_frame)
                )

        else:

            image = (
                image_path_or_frame
                .copy()
            )

        annotated_image = image.copy()

        if self.coco_model is None:
            print(
                "COCO model unavailable."
            )

            return (
                annotated_image,
                []
            )

        # -----------------------------------------------------
        # COCO DETECTION
        # -----------------------------------------------------

        try:

            results = self.coco_model(
                image,
                conf=0.25,
                verbose=False
            )[0]

        except Exception as e:

            print(
                "COCO detection error:",
                e
            )

            return (
                annotated_image,
                []
            )

        people = []
        motorcycles = []

        if results.boxes is not None:

            for box in results.boxes:

                try:

                    cls_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    coords = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                except Exception:
                    continue

                # COCO:
                # 0 = person
                # 3 = motorcycle

                if cls_id == 0:

                    people.append(
                        {
                            "box": coords,
                            "confidence": confidence
                        }
                    )

                elif cls_id == 3:

                    motorcycles.append(
                        {
                            "box": coords,
                            "confidence": confidence
                        }
                    )

        print(
            f"Detected people: {len(people)}"
        )

        print(
            f"Detected motorcycles: "
            f"{len(motorcycles)}"
        )

        # -----------------------------------------------------
        # RIDER ASSOCIATION
        # -----------------------------------------------------

        associations = (
            self.associate_riders_with_motorcycles(
                people,
                motorcycles
            )
        )

        print(
            "Rider associations:",
            associations
        )

        violations = []

        # -----------------------------------------------------
        # DRAW / PROCESS EACH MOTORCYCLE
        # -----------------------------------------------------

        for (
            motorcycle_index,
            rider_count,
            rider_indices
        ) in associations:

            motorcycle = motorcycles[
                motorcycle_index
            ]

            motorcycle_box = motorcycle[
                "box"
            ]

            mx1, my1, mx2, my2 = [
                int(v)
                for v in motorcycle_box
            ]

            # -------------------------------------------------
            # DRAW MOTORCYCLE
            # -------------------------------------------------

            cv2.rectangle(
                annotated_image,
                (mx1, my1),
                (mx2, my2),
                (255, 165, 0),
                2
            )

            cv2.putText(
                annotated_image,
                f"Bike: {rider_count} rider(s)",
                (
                    mx1,
                    max(20, my1 - 5)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 165, 0),
                2
            )

            # -------------------------------------------------
            # DRAW RIDERS
            # -------------------------------------------------

            for rider_index in rider_indices:

                rider_box = people[
                    rider_index
                ]["box"]

                rx1, ry1, rx2, ry2 = [
                    int(v)
                    for v in rider_box
                ]

                cv2.rectangle(
                    annotated_image,
                    (rx1, ry1),
                    (rx2, ry2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    annotated_image,
                    "Rider",
                    (
                        rx1,
                        max(20, ry1 - 5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            # -------------------------------------------------
            # MOTORCYCLE CROP
            # -------------------------------------------------

            motorcycle_crop = (
                self.get_motorcycle_crop(
                    image,
                    motorcycle_box
                )
            )

            # =================================================
            # TRIPLE RIDING
            # =================================================

            if rider_count == 3:

                print(
                    "TRIPLE RIDING DETECTED:"
                    f" {rider_count} riders"
                )

                cv2.rectangle(
                    annotated_image,
                    (mx1, my1),
                    (mx2, my2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    annotated_image,
                    "TRIPLE RIDING DETECTED",
                    (
                        mx1,
                        max(25, my1 - 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                evidence_path = (
                    self.save_evidence(
                        image,
                        "Triple Riding"
                    )
                )

                violations.append(
                    {
                        "violation_type":
                            "Triple Riding",

                        "fine":
                            1000,

                        "evidence_path":
                            evidence_path,

                        "motorcycle_index":
                            motorcycle_index,

                        "motorcycle_crop":
                            motorcycle_crop,

                        "rider_count":
                            rider_count,

                        "confidence":
                            float(
                                motorcycle[
                                    "confidence"
                                ]
                            )
                    }
                )

            # =================================================
            # NO HELMET
            # =================================================

            for rider_index in rider_indices:

                rider_box = people[
                    rider_index
                ]["box"]

                helmet_match = (
                    self.match_helmet_to_rider(
                        image,
                        rider_box
                    )
                )

                if helmet_match is None:
                    continue

                if (
                    helmet_match[
                        "class_name"
                    ]
                    !=
                    "Without Helmet"
                ):
                    continue

                # -------------------------------------------------
                # NO HELMET FOUND
                # -------------------------------------------------

                print(
                    "NO HELMET DETECTED "
                    f"for rider {rider_index}"
                )

                helmet_box = (
                    helmet_match["box"]
                )

                hx1, hy1, hx2, hy2 = [
                    int(v)
                    for v in helmet_box
                ]

                cv2.rectangle(
                    annotated_image,
                    (hx1, hy1),
                    (hx2, hy2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    annotated_image,
                    "NO HELMET",
                    (
                        hx1,
                        max(20, hy1 - 5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    annotated_image,
                    (mx1, my1),
                    (mx2, my2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    annotated_image,
                    "VIOLATION DETECTED",
                    (
                        mx1,
                        max(25, my1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                evidence_path = (
                    self.save_evidence(
                        image,
                        "Riding without Helmet"
                    )
                )

                violations.append(
                    {
                        "violation_type":
                            "Riding without Helmet",

                        "fine":
                            1000,

                        "evidence_path":
                            evidence_path,

                        "motorcycle_index":
                            motorcycle_index,

                        "motorcycle_crop":
                            motorcycle_crop,

                        "rider_index":
                            rider_index,

                        "rider_count":
                            rider_count,

                        "confidence":
                            float(
                                helmet_match[
                                    "confidence"
                                ]
                            )
                    }
                )

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        print(
            "Total violations detected:",
            len(violations)
        )

        return (
            annotated_image,
            violations
        )


if __name__ == "__main__":

    print(
        "TrafficViolationDetector "
        "module test"
    )

    detector = (
        TrafficViolationDetector()
    )

    print(
        "Detector initialized successfully."
    )