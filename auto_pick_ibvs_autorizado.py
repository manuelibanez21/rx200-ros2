#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTOPICK RX200 V2
Proyecto de tesis: enseñanza de rutas y percepción visual.

Incluye:
- Control manual de las cinco articulaciones del RX200.
- Apertura y cierre de la pinza.
- Enseñanza y almacenamiento de rutas.
- Carga y ejecución gradual de rutas.
- Detección de frutas mediante YOLO-seg.
- DroidCam como cámara global.
- RealSense D435i como cámara local.
- Lectura de profundidad.
- Diagnóstico visual IBVS usando el centro de la imagen DroidCam.
- Propuesta de rutas guardadas por clase y zona visual.
- Autorización explícita del usuario antes de ejecutar una ruta.

SEGURIDAD:
El IBVS de esta versión es SOLO DIAGNÓSTICO: calcula el error en píxeles,
pero NO convierte ese error en movimientos articulares. Para habilitar
correcciones reales se requiere calibrar la matriz de interacción/jacobiano,
validar límites y probar primero sin fruta ni agarre.
La detección YOLO no garantiza que un objeto sea una fruta.
"""

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.msg import JointSingleCommand
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(os.environ.get("RX200_MODEL_PATH", BASE_DIR / "fruit_yolo26s_seg" / "weights" / "best.pt"))
ROUTES_DIR = Path(os.environ.get("RX200_ROUTES_DIR", BASE_DIR / "rutas"))
CATALOG_PATH = ROUTES_DIR / "catalogo_rutas.json"

DROIDCAM_DEVICE = os.environ.get("RX200_DROIDCAM_DEVICE", "/dev/video8")

TOPIC_COLOR = "/camera/camera/color/image_raw"
TOPIC_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
TOPIC_INFO = "/camera/camera/color/camera_info"

TOPIC_ARM = "/rx200/commands/joint_group"
TOPIC_GRIPPER = "/rx200/commands/joint_single"

ROBOT_NAME = "rx200"

NUM_ARM_JOINTS = 5
CONFIDENCE_THRESHOLD = 0.75

# IBVS de diagnóstico en DroidCam. El objetivo inicial es el centro de imagen.
# No activar control correctivo sin calibrar la relación píxel-articulación.
IBVS_DEADZONE_PX = 12
IBVS_AUTO_CORRECTION = False  # Mantener False hasta calibración experimental.

MANUAL_STEP = 0.01
EXECUTION_STEP = 0.01
EXECUTION_PERIOD = 0.05

# Límites provisionales: comprobar con la configuración física.
JOINT_MIN = [-3.14, -2.0, -2.0, -2.0, -3.14]
JOINT_MAX = [3.14, 2.0, 2.0, 2.0, 3.14]

# Comprobar estos valores con el controlador de la pinza.
GRIPPER_OPEN_CMD = 200.0
GRIPPER_CLOSE_CMD = -200.0

# Solo se aceptan los nombres de las clases entrenadas.
ALLOWED_CLASSES = {"banana", "uvas", "fresa"}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def valid_number(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def safe_name(value):
    value = value.strip()
    return bool(value) and all(
        char.isalnum() or char in "_-"
        for char in value
    )


def depth_to_meters(depth_image, u, v):
    """Obtiene la mediana de profundidad de una región pequeña."""

    if depth_image is None:
        return None

    height, width = depth_image.shape[:2]

    if not (0 <= u < width and 0 <= v < height):
        return None

    radius = 2

    patch = depth_image[
        max(0, v - radius):min(height, v + radius + 1),
        max(0, u - radius):min(width, u + radius + 1)
    ]

    if patch.size == 0:
        return None

    valid = patch[np.isfinite(patch) & (patch > 0)]

    if valid.size == 0:
        return None

    value = float(np.median(valid))

    # RealSense 16UC1 expresa normalmente la profundidad en milímetros.
    if depth_image.dtype == np.uint16:
        value /= 1000.0

    if not math.isfinite(value) or value <= 0:
        return None

    return value


def detection_center(result, index):
    """
    Calcula el centro y el área a partir de la máscara de segmentación.
    Si no hay máscara válida, rechaza la detección.
    """

    if result.boxes is None or result.masks is None:
        return None

    if index < 0 or index >= len(result.boxes):
        return None

    if index >= len(result.masks.data):
        return None

    box = result.boxes[index]
    mask = result.masks.data[index].cpu().numpy() > 0.5

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    class_id = int(box.cls[0].item())
    confidence = float(box.conf[0].item())
    class_name = result.names.get(class_id, str(class_id))

    return {
        "class_id": class_id,
        "class_name": str(class_name).strip().lower(),
        "confidence": confidence,
        "cx": float(np.mean(xs)),
        "cy": float(np.mean(ys)),
        "area": float(np.count_nonzero(mask))
    }


# ============================================================
# NODO PRINCIPAL
# ============================================================

class AutoPickVisual(Node):

    def __init__(self):
        super().__init__("auto_pick_visual")

        self.bridge = CvBridge()

        # Cámaras
        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.last_color_time = None
        self.last_depth_time = None

        # Estado articular
        self.angulos = [0.0] * NUM_ARM_JOINTS

        # Estado del control
        self.ejecutando = False
        self.emergency_stop = False

        self.paso_manual = MANUAL_STEP
        self.paso_ejecucion = EXECUTION_STEP
        self.tiempo_ejecucion = EXECUTION_PERIOD

        self.limites_min = list(JOINT_MIN)
        self.limites_max = list(JOINT_MAX)

        # Ruta activa
        self.route = None
        self.route_name = None
        self.route_class = None
        self.route_approach = None

        # Selección de ruta y estado visual de DroidCam
        self.route_candidate_path = None
        self.route_candidate_data = None
        self.last_droid_target = None
        self.last_droid_frame = None
        self.last_droid_error = None
        self.last_droid_zone = None

        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

        # Publicadores del RX200
        self.pub_arm = self.create_publisher(
            JointGroupCommand,
            TOPIC_ARM,
            10
        )

        self.pub_gripper = self.create_publisher(
            JointSingleCommand,
            TOPIC_GRIPPER,
            10
        )

        # Suscripciones RealSense
        self.sub_color = self.create_subscription(
            Image,
            TOPIC_COLOR,
            self.color_callback,
            qos_profile_sensor_data
        )

        self.sub_depth = self.create_subscription(
            Image,
            TOPIC_DEPTH,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.sub_info = self.create_subscription(
            CameraInfo,
            TOPIC_INFO,
            self.info_callback,
            qos_profile_sensor_data
        )

        # Percepción
        self.last_droid_detections = []
        self.last_rs_detections = []

        self.ibvs_data = {
            "error_x_px": None,
            "error_y_px": None,
            "depth_m": None,
            "target_class": None
        }

        self.get_logger().info(
            "AutoPick V2 iniciado. IBVS en modo diagnóstico."
        )

    # ========================================================
    # CÁMARA REALSENSE
    # ========================================================

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
            self.last_color_time = time.time()

        except Exception as exc:
            self.get_logger().error(
                f"Error leyendo imagen RealSense: {exc}"
            )

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )
            self.last_depth_time = time.time()

        except Exception as exc:
            self.get_logger().error(
                f"Error leyendo profundidad: {exc}"
            )

    def info_callback(self, msg):
        self.camera_info = msg

    # ========================================================
    # YOLO
    # ========================================================

    def run_detection(self, model, frame):
        """Ejecuta YOLO-seg y devuelve la imagen anotada y las detecciones."""

        try:
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                task="segment",
                verbose=False
            )

            if results is None or len(results) == 0:
                return frame.copy(), []

            result = results[0]
            detections = []

            # Exigir cajas y máscaras. Sin máscara no se acepta detección.
            if result.boxes is None or result.masks is None:
                return frame.copy(), []

            if (
                len(result.boxes) == 0
                or len(result.masks.data) != len(result.boxes)
            ):
                return frame.copy(), []

            for i in range(len(result.boxes)):
                detection = detection_center(result, i)

                if detection is None:
                    continue

                if detection["confidence"] < CONFIDENCE_THRESHOLD:
                    continue

                if not (0 <= detection["cx"] < frame.shape[1]):
                    continue

                if not (0 <= detection["cy"] < frame.shape[0]):
                    continue

                detections.append(detection)

            return result.plot(), detections

        except Exception as exc:
            self.get_logger().error(f"Error en YOLO: {exc}")
            return frame.copy(), []

    @staticmethod
    def choose_target(detections):
        """
        Selecciona la detección con mayor confianza entre las clases conocidas.
        Esto no certifica que el objeto sea realmente una fruta.
        """

        if not detections:
            return None

        valid = []

        for detection in detections:
            if not detection:
                continue

            name = str(
                detection.get("class_name", "")
            ).strip().lower()

            try:
                confidence = float(
                    detection.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                continue

            if name not in ALLOWED_CLASSES:
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if not valid_number(detection.get("cx")):
                continue

            if not valid_number(detection.get("cy")):
                continue

            valid.append(detection)

        if not valid:
            return None

        return max(
            valid,
            key=lambda item: float(item["confidence"])
        )

    # ========================================================
    # DIAGNÓSTICO VISUAL IBVS
    # ========================================================

    def compute_ibvs_diagnostic(self, target, frame, depth_image):
        """
        Calcula el error en píxeles y consulta profundidad.
        No envía comandos al brazo.
        """

        if target is None:
            self.ibvs_data = {
                "error_x_px": None,
                "error_y_px": None,
                "depth_m": None,
                "target_class": None
            }
            return

        height, width = frame.shape[:2]

        center_x = width / 2.0
        center_y = height / 2.0

        cx = float(target["cx"])
        cy = float(target["cy"])

        u = int(round(cx))
        v = int(round(cy))

        depth_m = depth_to_meters(
            depth_image,
            u,
            v
        )

        self.ibvs_data = {
            "error_x_px": round(cx - center_x, 2),
            "error_y_px": round(cy - center_y, 2),
            "depth_m": (
                round(depth_m, 3)
                if depth_m is not None
                else None
            ),
            "target_class": target["class_name"]
        }

    # ========================================================
    # SELECCIÓN DE RUTA CON AUTORIZACIÓN EXPLÍCITA
    # ========================================================

    @staticmethod
    def visual_zone(target, frame):
        """Clasifica el centro visual en izquierda, centro o derecha."""
        if target is None or frame is None:
            return None
        width = frame.shape[1]
        cx = float(target.get("cx", -1))
        if width <= 0 or not (0 <= cx < width):
            return None
        ratio = cx / float(width)
        if ratio < 1.0 / 3.0:
            return "izquierda"
        if ratio > 2.0 / 3.0:
            return "derecha"
        return "centro"

    @staticmethod
    def normalize_zone(value):
        value = str(value or "").strip().lower()
        aliases = {
            "left": "izquierda", "izquierda": "izquierda",
            "middle": "centro", "center": "centro", "centre": "centro",
            "centro": "centro", "right": "derecha", "derecha": "derecha",
        }
        return aliases.get(value, value)

    def propose_route(self):
        """Busca una ruta guardada que coincida con clase y zona visual."""
        target = self.last_droid_target
        frame = getattr(self, "last_droid_frame", None)
        if target is None or frame is None:
            print("No hay una detección válida en DroidCam para proponer ruta.")
            return

        fruit_class = str(target.get("class_name", "")).strip().lower()
        if fruit_class not in ALLOWED_CLASSES:
            print("La clase detectada no está permitida.")
            return

        zone = self.visual_zone(target, frame)
        if zone is None:
            print("No se pudo determinar la zona visual.")
            return

        ROUTES_DIR.mkdir(parents=True, exist_ok=True)
        matches = []
        for path in sorted(ROUTES_DIR.glob("*.json")):
            if path.name == CATALOG_PATH.name:
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("trayectoria"), list):
                continue
            route_class = str(data.get("clase", "")).strip().lower()
            route_zone = self.normalize_zone(data.get("aproximacion", ""))
            # Se acepta también el nombre del archivo como indicio de zona.
            name_zone = self.normalize_zone(path.stem.lower().replace(fruit_class, ""))
            if route_class == fruit_class and (route_zone == zone or name_zone == zone):
                matches.append((path, data))

        if not matches:
            self.route_candidate_path = None
            self.route_candidate_data = None
            print(f"No hay ruta exacta para clase='{fruit_class}', zona='{zone}'.")
            print("No se ejecutará una ruta distinta automáticamente.")
            return

        if len(matches) == 1:
            chosen_path, chosen_data = matches[0]
        else:
            print("Hay varias rutas compatibles:")
            for i, (path, data) in enumerate(matches, start=1):
                print(f"{i}. {path.name} | aproximación={data.get('aproximacion')}")
            selection = input("Escribe el número de la ruta que deseas proponer (0 cancela): ").strip()
            try:
                idx = int(selection)
            except ValueError:
                print("Selección inválida; no se propone ninguna ruta.")
                return
            if idx == 0 or not 1 <= idx <= len(matches):
                print("Selección cancelada.")
                return
            chosen_path, chosen_data = matches[idx - 1]

        self.route_candidate_path = chosen_path
        self.route_candidate_data = chosen_data
        print("\n========== RUTA PROPUESTA ==========")
        print(f"Fruta detectada: {fruit_class}")
        print(f"Confianza YOLO: {float(target.get('confidence', 0.0)):.2f}")
        print(f"Zona de imagen: {zone}")
        print(f"Ruta propuesta: {chosen_path.name}")
        print("Esta propuesta NO es una verificación de identidad ni de seguridad.")
        print("Pulsa U para solicitar autorización o N para cancelar.")
        print("===================================\n")

    def authorize_candidate_route(self):
        """Carga la ruta propuesta y exige confirmación escrita antes de mover."""
        if self.route_candidate_path is None or self.route_candidate_data is None:
            print("Primero pulsa C para proponer una ruta compatible.")
            return

        data = self.route_candidate_data
        trajectory = data.get("trayectoria", [])
        arm_actions = [a for a in trajectory if isinstance(a, dict) and a.get("tipo") == "arm"]
        if not arm_actions:
            print("Ruta rechazada: no contiene posiciones articulares.")
            return

        print("\nANTES DE AUTORIZAR: verifica el espacio de trabajo, la posición real del brazo, "
              "los límites articulares y que la ruta haya sido probada sin fruta.")
        confirm = input(f"Para cargar {self.route_candidate_path.name}, escribe CARGAR: ").strip()
        if confirm != "CARGAR":
            print("Carga cancelada.")
            return

        self.route = data
        self.route_name = data.get("nombre", self.route_candidate_path.stem)
        self.route_class = data.get("clase")
        self.route_approach = data.get("aproximacion")
        print(f"Ruta cargada: {self.route_candidate_path.name}")
        print("La carga no mueve el brazo. Para ejecutar, usa E y confirma EJECUTAR.")

    def compute_droidcam_ibvs_diagnostic(self, target, frame):
        """Calcula error visual en píxeles. No emite comandos articulares."""
        if target is None or frame is None:
            self.last_droid_error = None
            self.last_droid_zone = None
            return
        height, width = frame.shape[:2]
        desired_x, desired_y = width / 2.0, height / 2.0
        cx, cy = float(target["cx"]), float(target["cy"])
        self.last_droid_error = {
            "error_x_px": round(cx - desired_x, 2),
            "error_y_px": round(cy - desired_y, 2),
            "target_x": round(desired_x, 2),
            "target_y": round(desired_y, 2),
        }
        self.last_droid_zone = self.visual_zone(target, frame)

    def draw_droidcam_ibvs(self, image, target):
        """Dibuja el punto deseado, centro detectado y error para diagnóstico."""
        h, w = image.shape[:2]
        desired = (w // 2, h // 2)
        cv2.drawMarker(image, desired, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if target is not None:
            actual = (int(round(target["cx"])), int(round(target["cy"])))
            cv2.circle(image, actual, 5, (0, 255, 0), -1)
            cv2.line(image, actual, desired, (0, 255, 0), 2)
        if self.last_droid_error is not None:
            err = self.last_droid_error
            label = f"IBVS diagnostico ex={err['error_x_px']} ey={err['error_y_px']} px"
            cv2.putText(image, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(image, "SIN CORRECCION AUTOMATICA", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        if self.route_candidate_path is not None:
            cv2.putText(image, f"Ruta propuesta: {self.route_candidate_path.name}", (10, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.putText(image, "C: proponer | U: cargar con autorizacion | E: ejecutar con confirmacion",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        return image

    # ========================================================
    # CONTROL MANUAL
    # ========================================================

    def publish_arm_command(self):
        if self.emergency_stop:
            print("Parada lógica activa: comando de brazo bloqueado.")
            return

        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [float(value) for value in self.angulos]

        self.pub_arm.publish(msg)

    def change_joint(self, index, delta):
        if self.emergency_stop:
            print("Parada lógica activa. Reinicia el nodo para continuar.")
            return

        if not 0 <= index < NUM_ARM_JOINTS:
            return

        new_value = self.angulos[index] + delta

        new_value = max(
            self.limites_min[index],
            min(self.limites_max[index], new_value)
        )

        self.angulos[index] = new_value

        self.publish_arm_command()

        print(
            f"Articulación {index + 1}: "
            f"{self.angulos[index]:.3f} rad"
        )

    def move_gripper(self, open_gripper):
        if self.emergency_stop:
            print("Parada lógica activa: pinza bloqueada.")
            return

        msg = JointSingleCommand()
        msg.name = "gripper"

        if open_gripper:
            msg.cmd = float(GRIPPER_OPEN_CMD)
            print("Orden de apertura de pinza enviada.")
        else:
            msg.cmd = float(GRIPPER_CLOSE_CMD)
            print("Orden de cierre de pinza enviada.")

        self.pub_gripper.publish(msg)

    # ========================================================
    # ENSEÑANZA DE RUTAS
    # ========================================================

    def create_route(self):
        if self.emergency_stop:
            print("Parada lógica activa.")
            return

        name = input(
            "Nombre de la ruta (letras, números, _ o -): "
        ).strip()

        if not safe_name(name):
            print("Nombre no válido.")
            return

        fruit_class = input(
            "Clase asociada (banana, uvas o fresa): "
        ).strip().lower()

        if fruit_class not in ALLOWED_CLASSES:
            print("Clase no válida.")
            return

        approach = input(
            "Descripción de aproximación: "
        ).strip()

        self.route = {
            "nombre": name,
            "clase": fruit_class,
            "aproximacion": approach,
            "creada": datetime.now().isoformat(timespec="seconds"),
            "trayectoria": []
        }

        self.route_name = name
        self.route_class = fruit_class
        self.route_approach = approach

        print(f"Ruta '{name}' creada.")
        print("Mueve el brazo manualmente y pulsa ENTER para registrar posiciones.")

    def add_current_arm_action(self):
        if self.route is None:
            print("Primero crea una ruta con N.")
            return

        if self.emergency_stop:
            print("Parada lógica activa.")
            return

        action = {
            "tipo": "arm",
            "angulos": [
                float(value) for value in self.angulos
            ],
            "tiempo": datetime.now().isoformat(timespec="seconds")
        }

        self.route["trayectoria"].append(action)

        print(
            f"Posición registrada. "
            f"Total de acciones: {len(self.route['trayectoria'])}"
        )

    def save_visual_observation(self):
        if self.route is None:
            print("Primero crea una ruta con N.")
            return

        detections = self.last_rs_detections

        if not detections:
            print("No hay detecciones de RealSense para registrar.")
            return

        target = self.choose_target(detections)

        if target is None:
            print("No hay una detección válida de una clase permitida.")
            return

        action = {
            "tipo": "observacion_visual",
            "clase": target["class_name"],
            "confianza": float(target["confidence"]),
            "centro_x": float(target["cx"]),
            "centro_y": float(target["cy"]),
            "profundidad_m": self.ibvs_data.get("depth_m"),
            "tiempo": datetime.now().isoformat(timespec="seconds")
        }

        self.route["trayectoria"].append(action)

        print(
            "Observación visual registrada. "
            "No implica que el objeto haya sido verificado físicamente."
        )

    def save_route(self):
        if self.route is None:
            print("No hay una ruta activa para guardar.")
            return

        if not self.route["trayectoria"]:
            print("La ruta no tiene acciones registradas.")
            return

        route_name = self.route["nombre"]

        if not safe_name(route_name):
            print("Nombre de ruta no válido.")
            return

        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

        route_path = ROUTES_DIR / f"{route_name}.json"

        if route_path.exists():
            confirm = input(
                f"La ruta '{route_name}' ya existe. "
                "¿Sobrescribir? Escribe SI: "
            ).strip()

            if confirm != "SI":
                print("Guardado cancelado.")
                return

        try:
            with route_path.open("w", encoding="utf-8") as file:
                json.dump(
                    self.route,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

            self.update_catalog(route_name)

            print(f"Ruta guardada en: {route_path}")

        except OSError as exc:
            print(f"Error guardando ruta: {exc}")

    def update_catalog(self, route_name):
        catalog = []

        if CATALOG_PATH.exists():
            try:
                with CATALOG_PATH.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)

                if isinstance(loaded, list):
                    catalog = loaded

            except (OSError, json.JSONDecodeError):
                catalog = []

        if route_name not in catalog:
            catalog.append(route_name)

        try:
            with CATALOG_PATH.open("w", encoding="utf-8") as file:
                json.dump(
                    catalog,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

        except OSError as exc:
            print(f"No se pudo actualizar el catálogo: {exc}")

    def load_route(self):
        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

        files = sorted(ROUTES_DIR.glob("*.json"))
        files = [
            path for path in files
            if path.name != CATALOG_PATH.name
        ]

        if not files:
            print("No hay rutas guardadas.")
            return

        print("\nRutas disponibles:")

        for index, path in enumerate(files, start=1):
            print(f"{index}. {path.stem}")

        selection = input(
            "Número de ruta para cargar: "
        ).strip()

        try:
            index = int(selection) - 1
        except ValueError:
            print("Selección no válida.")
            return

        if not 0 <= index < len(files):
            print("Selección fuera de rango.")
            return

        try:
            with files[index].open("r", encoding="utf-8") as file:
                route = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error cargando la ruta: {exc}")
            return

        if not isinstance(route, dict):
            print("El archivo de ruta no tiene un formato válido.")
            return

        trajectory = route.get("trayectoria")

        if not isinstance(trajectory, list):
            print("La trayectoria de la ruta no es válida.")
            return

        self.route = route
        self.route_name = route.get("nombre", files[index].stem)
        self.route_class = route.get("clase")
        self.route_approach = route.get("aproximacion")

        print(f"Ruta cargada: {self.route_name}")
        print(f"Clase asociada: {self.route_class}")
        print(f"Acciones: {len(trajectory)}")

    def execute_route(self):
        if self.emergency_stop:
            print("Parada lógica activa. No se puede ejecutar una ruta.")
            return

        if self.route is None:
            print("No hay una ruta cargada.")
            return

        trajectory = self.route.get("trayectoria", [])

        arm_actions = [
            action for action in trajectory
            if isinstance(action, dict)
            and action.get("tipo") == "arm"
        ]

        if not arm_actions:
            print("La ruta no contiene posiciones articulares.")
            return

        confirm = input(
            "ADVERTENCIA: el brazo se moverá siguiendo la ruta. "
            "Comprueba que el espacio esté despejado. "
            "Escribe EJECUTAR para continuar: "
        ).strip()

        if confirm != "EJECUTAR":
            print("Ejecución cancelada.")
            return

        self.ejecutando = True

        try:
            for action in arm_actions:
                if self.emergency_stop:
                    print("Parada lógica activada durante la ruta.")
                    break

                angles = action.get("angulos")

                if not isinstance(angles, list):
                    print("Acción ignorada: falta la lista de ángulos.")
                    continue

                if len(angles) != NUM_ARM_JOINTS:
                    print("Acción ignorada: número de articulaciones incorrecto.")
                    continue

                try:
                    target = [float(value) for value in angles]
                except (TypeError, ValueError):
                    print("Acción ignorada: hay valores no numéricos.")
                    continue

                if not all(math.isfinite(value) for value in target):
                    print("Acción ignorada: hay ángulos no finitos.")
                    continue

                if any(
                    target[i] < self.limites_min[i]
                    or target[i] > self.limites_max[i]
                    for i in range(NUM_ARM_JOINTS)
                ):
                    print("Acción ignorada: ángulos fuera de los límites configurados.")
                    continue

                start = list(self.angulos)
                max_delta = max(
                    abs(target[i] - start[i])
                    for i in range(NUM_ARM_JOINTS)
                )

                steps = max(
                    1,
                    int(math.ceil(max_delta / self.paso_ejecucion))
                )

                for step in range(1, steps + 1):
                    if self.emergency_stop:
                        break

                    ratio = step / steps

                    self.angulos = [
                        start[i] + (target[i] - start[i]) * ratio
                        for i in range(NUM_ARM_JOINTS)
                    ]

                    self.publish_arm_command()
                    time.sleep(self.tiempo_ejecucion)

                if self.emergency_stop:
                    break

            print("Ejecución de ruta finalizada.")

        finally:
            self.ejecutando = False

    def list_routes(self):
        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

        files = sorted(ROUTES_DIR.glob("*.json"))
        files = [
            path for path in files
            if path.name != CATALOG_PATH.name
        ]

        if not files:
            print("No hay rutas guardadas.")
            return

        print("\n========== RUTAS GUARDADAS ==========")

        for path in files:
            try:
                with path.open("r", encoding="utf-8") as file:
                    route = json.load(file)

                print(
                    f"{path.stem} | "
                    f"fruta={route.get('clase')} | "
                    f"aproximación={route.get('aproximacion')} | "
                    f"acciones={len(route.get('trayectoria', []))}"
                )

            except (OSError, json.JSONDecodeError) as exc:
                print(f"Error leyendo {path.name}: {exc}")

        print("=====================================")

    def discard_route(self):
        if self.route is None:
            print("No hay una ruta activa.")
            return

        confirm = input(
            "¿Descartar la ruta sin guardarla? Escribe SI: "
        ).strip()

        if confirm == "SI":
            self.route = None
            self.route_name = None
            self.route_class = None
            self.route_approach = None
            print("Ruta descartada.")

    # ========================================================
    # INFORMACIÓN SOBRE LAS IMÁGENES
    # ========================================================

    def draw_status(
        self,
        image,
        title,
        detections,
        is_realsense=False
    ):
        height, width = image.shape[:2]

        cv2.drawMarker(
            image,
            (width // 2, height // 2),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2
        )

        cv2.putText(
            image,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Detecciones: {len(detections)}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        if is_realsense:
            data = self.ibvs_data

            cv2.putText(
                image,
                f"Error X: {data['error_x_px']}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                image,
                f"Error Y: {data['error_y_px']}",
                (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                image,
                f"Profundidad: {data['depth_m']} m",
                (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                image,
                "IBVS: SOLO DIAGNOSTICO",
                (10, 154),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        return image

    # ========================================================
    # BUCLE PRINCIPAL
    # ========================================================

    def run(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encuentra el modelo: {MODEL_PATH}"
            )

        print(f"Cargando YOLO: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))

        print(f"Abriendo DroidCam: {DROIDCAM_DEVICE}")

        droidcam = cv2.VideoCapture(
            DROIDCAM_DEVICE,
            cv2.CAP_V4L2
        )

        if not droidcam.isOpened():
            raise RuntimeError(
                f"No se pudo abrir {DROIDCAM_DEVICE}."
            )

        droidcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        droidcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n========== CONTROLES ==========")
        print("A/D: articulación 1")
        print("W/S: articulación 2")
        print("R/F: articulación 3")
        print("T/G: articulación 4")
        print("Y/H: articulación 5")
        print("Z: abrir pinza | V: cerrar pinza")
        print("N: crear ruta")
        print("ENTER: guardar posición actual en la ruta")
        print("B: guardar observación visual")
        print("P: guardar ruta")
        print("L: cargar ruta")
        print("E: ejecutar ruta cargada (pide confirmación escrita)")
        print("C: proponer ruta según fruta y zona visual")
        print("U: cargar ruta propuesta (pide autorización escrita)")
        print("I: listar rutas")
        print("X: descartar ruta")
        print("ESC: parada lógica")
        print("AVISO: la detección visual no confirma la identidad real del objeto.")
        print("IBVS calcula error en píxeles, pero NO corrige automáticamente articulaciones.")
        print("Prueba rutas primero sin fruta y con zona de movimiento despejada.")
        print("Q: salir")
        print("===============================\n")

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)

                # --------------------------------------------
                # DROIDCAM: VISTA GLOBAL
                # --------------------------------------------

                ok, droid_frame = droidcam.read()

                if ok:
                    droid_annotated, detections = self.run_detection(
                        model,
                        droid_frame
                    )

                    self.last_droid_detections = detections
                    self.last_droid_frame = droid_frame.copy()

                    target = self.choose_target(detections)
                    self.last_droid_target = target
                    self.compute_droidcam_ibvs_diagnostic(target, droid_frame)

                    droid_annotated = self.draw_status(
                        droid_annotated,
                        "DROIDCAM - VISTA GLOBAL",
                        detections
                    )

                    droid_annotated = self.draw_droidcam_ibvs(droid_annotated, target)

                    if target is not None:
                        cv2.putText(
                            droid_annotated,
                            f"Objetivo: {target['class_name']}",
                            (10, droid_annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                    cv2.imshow(
                        "AutoPick - DroidCam",
                        droid_annotated
                    )

                # --------------------------------------------
                # REALSENSE: VISTA LOCAL
                # --------------------------------------------

                if self.color_image is not None:
                    rs_frame = self.color_image.copy()

                    rs_annotated, detections = self.run_detection(
                        model,
                        rs_frame
                    )

                    self.last_rs_detections = detections

                    target = self.choose_target(detections)

                    self.compute_ibvs_diagnostic(
                        target,
                        rs_frame,
                        self.depth_image
                    )

                    rs_annotated = self.draw_status(
                        rs_annotated,
                        "REALSENSE - VISTA LOCAL",
                        detections,
                        is_realsense=True
                    )

                    cv2.imshow(
                        "AutoPick - RealSense IBVS",
                        rs_annotated
                    )

                # --------------------------------------------
                # TECLADO
                # --------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q")):
                    break

                elif key == 27:
                    self.emergency_stop = True
                    print(
                        "Parada lógica activada. "
                        "No sustituye la parada física."
                    )

                elif key in (ord("n"), ord("N")):
                    self.create_route()

                elif key in (13, 10):
                    self.add_current_arm_action()

                elif key in (ord("z"), ord("Z")):
                    self.move_gripper(True)

                elif key in (ord("v"), ord("V")):
                    self.move_gripper(False)

                elif key in (ord("b"), ord("B")):
                    self.save_visual_observation()

                elif key in (ord("p"), ord("P")):
                    self.save_route()

                elif key in (ord("l"), ord("L")):
                    self.load_route()

                elif key in (ord("e"), ord("E")):
                    self.execute_route()

                elif key in (ord("c"), ord("C")):
                    self.propose_route()

                elif key in (ord("u"), ord("U")):
                    self.authorize_candidate_route()

                elif key in (ord("i"), ord("I")):
                    self.list_routes()

                elif key in (ord("x"), ord("X")):
                    self.discard_route()

                elif key in (ord("a"), ord("A")):
                    self.change_joint(0, -self.paso_manual)

                elif key in (ord("d"), ord("D")):
                    self.change_joint(0, self.paso_manual)

                elif key in (ord("w"), ord("W")):
                    self.change_joint(1, self.paso_manual)

                elif key in (ord("s"), ord("S")):
                    self.change_joint(1, -self.paso_manual)

                elif key in (ord("r"), ord("R")):
                    self.change_joint(2, self.paso_manual)

                elif key in (ord("f"), ord("F")):
                    self.change_joint(2, -self.paso_manual)

                elif key in (ord("t"), ord("T")):
                    self.change_joint(3, self.paso_manual)

                elif key in (ord("g"), ord("G")):
                    self.change_joint(3, -self.paso_manual)

                elif key in (ord("y"), ord("Y")):
                    self.change_joint(4, self.paso_manual)

                elif key in (ord("h"), ord("H")):
                    self.change_joint(4, -self.paso_manual)

        finally:
            droidcam.release()
            cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = AutoPickVisual()

    try:
        node.run()

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print(f"\nERROR: {exc}")

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
