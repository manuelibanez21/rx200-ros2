#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
import cv2
import numpy as np
import time


class AutoPickRed(Node):

    def __init__(self):
        super().__init__('auto_pick_red')

        # Publicadores del brazo
        self.pub_group = self.create_publisher(
            JointGroupCommand,
            '/rx200/commands/joint_group',
            10
        )

        self.pub_single = self.create_publisher(
            JointSingleCommand,
            '/rx200/commands/joint_single',
            10
        )

        # Cámara IP (iPhone)
        self.declare_parameter("camera_url", os.environ.get("RX200_CAMERA_URL", ""))
        camera_url = self.get_parameter("camera_url").value
        if not camera_url:
            raise ValueError("Configura camera_url o RX200_CAMERA_URL para la cámara IP.")
        self.cap = cv2.VideoCapture(camera_url)

        cv2.namedWindow("Brazo - Cámara", cv2.WINDOW_NORMAL)

        # Variables del brazo
        self.angulos = [0.0]*5   # waist, shoulder, elbow, wrist_angle, wrist_rotate
        self.incremento = 0.03  # movimiento suave

        self.run()


    # ===================== PUBLICAR BRAZO =====================

    def publicar_brazo(self):
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = self.angulos
        self.pub_group.publish(msg)


    def publicar_garra_pwm(self, pwm):
        msg = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd = float(pwm)
        self.pub_single.publish(msg)


    # ===================== MOVIMIENTO =====================

    def move_towards(self, center, frame_shape):

        frame_center = (
            frame_shape[1] // 2,
            frame_shape[0] // 2
        )

        dx = center[0] - frame_center[0]
        dy = center[1] - frame_center[1]

        # Ajustes suaves
        self.angulos[0] += np.clip(dx / 1500, -self.incremento, self.incremento)
        self.angulos[1] -= np.clip(dy / 2500, -self.incremento, self.incremento)
        self.angulos[2] += np.clip(dy / 4000, -self.incremento, self.incremento)

        self.angulos[3] = 0.0
        self.angulos[4] = 0.0

        self.publicar_brazo()


    # ===================== LOOP PRINCIPAL =====================

    def run(self):

        while rclpy.ok():

            ret, frame = self.cap.read()
            if not ret:
                continue

            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 180, 100])
            upper_red1 = np.array([10, 255, 255])

            lower_red2 = np.array([160, 180, 100])
            upper_red2 = np.array([179, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:

                largest = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest) > 300:

                    x, y, w, h = cv2.boundingRect(largest)
                    center = (x + w//2, y + h//2)

                    self.move_towards(center, frame.shape)

                    # Agarre cuando está cerca
                    if w * h > 8000:
                        self.publicar_garra_pwm(-200)   # cerrar
                    else:
                        self.publicar_garra_pwm(200)    # abrir

                    # Dibujos
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.circle(frame, center, 5, (255,0,0), -1)

            cv2.imshow("Brazo - Cámara", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

        self.cap.release()
        cv2.destroyAllWindows()


# ===================== MAIN =====================

def main():
    rclpy.init()
    nodo = AutoPickRed()
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
