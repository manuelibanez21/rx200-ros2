cmake_minimum_required(VERSION 3.18)
project(rx200_ros2)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  auto_pick_rx200_completo_supervisado.py
  auto_pick_ibvs_autorizado.py
  auto_pick_lento_v2.py
  control_teclado_naranja.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES hri_bridge.py DESTINATION lib/${PROJECT_NAME})
install(FILES rutas.zip trayectoria_rx200.json DESTINATION share/${PROJECT_NAME})
file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/rutas.zip"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/route_data")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/route_data/rutas"
        DESTINATION lib/${PROJECT_NAME})

ament_package()
