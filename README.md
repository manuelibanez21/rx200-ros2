# RX200 con ROS 2

Paquete de manipulación supervisada para Interbotix RX200. Incluye los cuatro programas solicitados, el ZIP original, las rutas que se extraen al compilar y la trayectoria independiente.

## Compilar

Usa Linux con ROS 2, `colcon` y el workspace del controlador Interbotix disponible. Instala Ultralytics en el mismo intérprete que usa ROS 2, de modo que también pueda importar `rclpy` y `cv_bridge`. El peso YOLO de segmentación `best.pt` debe proporcionarse aparte.

```bash
source /opt/ros/"$ROS_DISTRO"/setup.bash
# Si el controlador Interbotix está en otro workspace, carga aquí su install/setup.bash.
# Coloca este repositorio como ~/ros2_ws/src/rx200-ros2.
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select rx200_ros2
source install/setup.bash
```

Arranca el driver RX200 y RealSense con profundidad alineada al color. Los perfiles de suscripción de imagen aceptan publicadores de sensores best effort. La versión supervisada exige `/rx200/joint_states` reciente para el movimiento. Ejecuta una sola variante de control a la vez.

```bash
export RX200_MODEL_PATH=/ruta/al/best.pt
export RX200_DROIDCAM_DEVICE=/dev/video8
# Opcional. Extrae rutas.zip en una carpeta de trabajo para editar las rutas.
# export RX200_ROUTES_DIR="$HOME/rx200-datos/rutas"
# export RX200_IBVS_JACOBIAN_PATH="$RX200_ROUTES_DIR/ibvs_jacobian.json"

ros2 run rx200_ros2 auto_pick_rx200_completo_supervisado.py --ros-args -p enable_hri:=true
```

Las variantes `auto_pick_ibvs_autorizado.py` y `auto_pick_lento_v2.py` admiten las mismas variables para modelo, rutas y DroidCam. Para el teclado IP, configura la URL de la cámara antes de arrancar:

```bash
ros2 run rx200_ros2 control_teclado_naranja.py --ros-args -p camera_url:=http://DIRECCION:4747/video.mjpeg
```

Los tópicos del driver y la cámara se pueden remapear con `--ros-args -r tópico_original:=tópico_real`. Si usas namespaces, confirma sus nombres con `ros2 topic list -t`.

## Integración

| Componente | Interfaz y alcance |
| --- | --- |
| HRI | `/vision/buscar_objeto` recibe `std_msgs/msg/String`. Ante una detección fresca de la clase pedida con profundidad válida, publica `/detected_object` (`String`) y `/object_position` (`geometry_msgs/msg/Point`, metros en el marco óptico de color). |
| HRI | `/garra/agarrar` (`String`) y `/garra/entregar` (`Bool`) muestran solicitudes en la consola. Tras completar la operación con los controles locales, pulsa **M** y escribe **CONFIRMAR** tras verificar el resultado, o **RECHAZAR**. Se publica el resultado en `/garra/tiene_objeto` o `/garra/entregado`, respectivamente. HRI espera 60 s para agarre y 30 s para entrega. |
| Computer Vision | Los nombres y tipos publicados coinciden con `Dataset/Code_Dataset/detect.py`. Usa YOLO-seg integrado como fuente de frutas, o ejecuta el detector externo en tópicos remapeados para evitar que dos nodos publiquen respuestas en el mismo contrato. |
| Navegation | Arranca el stack Nav2 del repositorio con la acción `/nav/navigate_to_pose` (`nav2_msgs/action/NavigateToPose`). HRI es el cliente de esa acción y coordina la navegación con la manipulación. |
| martha_nav | El planificador PPO tiene su propio contrato de base. Recibe `/goal_pose` (`PoseStamped`) y `/cancel_goal` (`Empty`), publica `/nav_status` (`String`) y controla `/cmd_vel`. Puede operar junto al RX200 con coordinación manual. Para la secuencia automática HRI, usa el servidor Nav2 de Navegation o implementa y valida un adaptador de acciones específico para Martha. Utiliza una sola fuente de mando para la base. |

El modelo de frutas admite `banana`, `uvas` y `fresa`. La tarea de bolsa (`/garra/proceder`) requiere su propio detector y procedimiento. El registro y reconocimiento de personas corresponden a los nodos de visión de HRI. Arranca el orquestador con la configuración real de equipos y deshabilita su simulador de agarre y visión.

Los mensajes del orquestador dejan solicitudes pendientes para el operador. Usa los controles existentes para seleccionar la zona, cargar y ejecutar la ruta autorizada. El movimiento y las correcciones IBVS conservan sus confirmaciones locales. La ejecución de una trayectoria no verifica que la fruta quedó en la pinza.

## Comprobación

Con el driver de motores apagado, verifica el contrato del detector y confirma que aparece una solicitud en consola:

```bash
ros2 topic info -v /vision/buscar_objeto
ros2 topic echo /detected_object  # En otra terminal.
ros2 topic pub --once /vision/buscar_objeto std_msgs/msg/String "{data: fresa}"
ros2 topic pub --once /garra/agarrar std_msgs/msg/String "{data: fresa}"
ros2 action info /nav/navigate_to_pose
```

Comprueba que RealSense publica color, profundidad alineada y calibración del mismo marco óptico. La búsqueda usa imágenes posteriores a la solicitud y exige marcas de color y profundidad con diferencia máxima de 50 ms. HRI espera un punto y una clase en tópicos separados. El punto carece de cabecera por compatibilidad con ese contrato.

`trayectoria_rx200.json` se incluye como trayectoria histórica. El selector supervisado usa los archivos de `rutas/`.

## Repositorios de referencia

- [Navegation](https://github.com/AndesRobot-Oficial/Navegation)
- [Human-Robot-Interaction](https://github.com/AndesRobot-Oficial/Human-Robot-Interaction)
- [Computer_Vision](https://github.com/AndesRobot-Oficial/Computer_Vision)
- [martha_nav](https://github.com/Mocolax/martha_nav)
