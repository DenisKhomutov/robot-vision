{
  "move_type": "straight",       // "straight" | "left" | "right" | "stop" | "lost"
  "deg": 3.42,                   // угол, только для straight/left/right
  "offset": 0.012, "dist_to_route": 0.03,
  "offset_m": 0.05, "dist_to_route_m": 0.12,   // метры, если известен масштаб карты; иначе null
  "node": 87, "target_node": 92,               // ЛОКАЛЬНЫЙ узел активного шарда
  "global_node": 774, "global_target_node": 779, // сквозной узел по всему маршруту (шардинг)
  "inliers": 63,
  "pos": [x, z], "head": [x, z],  // позиция и курс в плоскости карты
  "paused": true,                 // есть ТОЛЬКО когда на паузе; в резюме поле отсутствует
  "reason": "PnP не сошёлся",     // есть только при lost
  "cam": "front", "map": "route12_front_bal9_03_of_09",
  "mode": "dual", "route": "route12",
  "full_map_recovery": true, "recovery_map": "route12_front_colmap_full_...",  // есть только в кадре восстановления
  "map_mismatch": true,           // есть только когда front/rear карты разных маршрутов и фронт потерян
  "traffic_enabled": false, "traffic_loading": false,
  "traffic_in_zone": false, "traffic_state": "WAIT_RED",
  "ts": 1787052345.123
}
