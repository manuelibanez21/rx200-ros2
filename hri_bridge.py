"""Interfaz de percepción y confirmación humana del orquestador HRI."""
import time

from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String


class HriBridge:
    def __init__(self, node, allowed_classes):
        self.allowed_classes = allowed_classes
        self.search = None
        self.request = None
        self.object_pub = node.create_publisher(String, "/detected_object", 10)
        self.point_pub = node.create_publisher(Point, "/object_position", 10)
        self.held_pub = node.create_publisher(Bool, "/garra/tiene_objeto", 10)
        self.delivered_pub = node.create_publisher(Bool, "/garra/entregado", 10)
        node.create_subscription(String, "/vision/buscar_objeto", self.on_search, 10)
        node.create_subscription(String, "/garra/agarrar", self.on_grasp, 10)
        node.create_subscription(Bool, "/garra/proceder", self.on_proceed, 10)
        node.create_subscription(Bool, "/garra/entregar", self.on_delivery, 10)

    def on_search(self, msg):
        label = msg.data.strip().lower()
        self.search = (msg.data, label, time.monotonic()) if label in self.allowed_classes else None

    def publish_detection(self, detections, frame_received, project):
        if self.search is None:
            return
        requested, label, received = self.search
        if time.monotonic() - received >= 30.0:
            self.search = None
            return
        if frame_received <= received:
            return

        matches = sorted(
            (item for item in detections if item["class_name"].lower() == label),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        for item in matches:
            point = project(item)
            if point is not None:
                self.point_pub.publish(Point(x=point[0], y=point[1], z=point[2]))
                self.object_pub.publish(String(data=requested))
                self.search = None
                break

    def on_grasp(self, msg):
        label = msg.data.strip().lower()
        if label in self.allowed_classes:
            self.queue_request(f"agarre de {label}", self.held_pub, 60.0)

    def on_proceed(self, msg):
        if msg.data:
            print("HRI solicitó proceder con la bolsa. Este modelo sólo admite frutas.")

    def on_delivery(self, msg):
        if msg.data:
            self.queue_request("entrega", self.delivered_pub, 30.0)

    def queue_request(self, description, publisher, timeout):
        if self.request is not None and time.monotonic() < self.request[2]:
            print(f"HRI: solicitud de {description} omitida por una solicitud pendiente.")
            return
        self.request = (description, publisher, time.monotonic() + timeout)
        print(f"HRI: solicitud de {description}. Completa la operación local y pulsa M para informar el resultado.")

    def report_result(self):
        request = self.request
        self.request = None
        if request is None or time.monotonic() >= request[2]:
            print("HRI: no hay una solicitud vigente.")
            return
        description, publisher, deadline = request
        answer = input(f"HRI: comprueba físicamente el resultado de {description}. Escribe CONFIRMAR o RECHAZAR: ")
        if time.monotonic() >= deadline:
            print("HRI: la solicitud venció.")
        elif answer.strip() in {"CONFIRMAR", "RECHAZAR"}:
            publisher.publish(Bool(data=answer.strip() == "CONFIRMAR"))
        else:
            self.request = request
