"""
Детекция динамических препятствий — Hokuyo UTM-30LX EW.

Управление: Q — выход, R — сброс треков, +/- — масштаб, SPACE — пауза.
"""

import cv2
import numpy as np
import socket
import math
import time
from collections import deque
from datetime import datetime

#  НАСТРОЙКИ

LIDAR_IP   = "192.168.0.10"
LIDAR_PORT = 10940

# Лидар
NUM_POINTS   = 1080
ANGLE_START  = -135.0   # градусы
ANGLE_RANGE  = 270.0    # градусы
DIST_MIN_MM  = 100      # ближе — шум/мёртвая зона лидара
DIST_MAX_MM  = 15000    # дальше — игнорируем (реальный предел в помещении)
FRAME_DT     = 0.05     # секунд между кадрами

# Кластеризация
CLUSTER_SPLIT_MM    = 200  # мм — разрыв между соседними точками → новый кластер
CLUSTER_MIN_PTS     = 25   # минимум точек — отсекаем мелкий шум
CLUSTER_MAX_PTS     = 200  # максимум (стены и пол отсекаем)
CLUSTER_MIN_SIZE_MM = 50   # минимальный размер кластера по любой оси (мм)
CLUSTER_MAX_SIZE_MM = 1500 # максимальный размер кластера по любой оси (мм) — стены
CLUSTER_MAX_ASPECT  = 6.0  # макс. соотношение сторон bounding box — стены вытянуты

# Трекинг
MATCH_DIST_MM     = 600  # мм — макс. дистанция сопоставления кластера с треком
MAX_LOST_FRAMES   = 8    # кадров до удаления потерянного трека
MIN_AGE_DYNAMIC   = 8    # минимум кадров наблюдения до вердикта "динамический"
TRAIL_LEN         = 50   # длина следа траектории (кадров)

# Детекция движения
DYNAMIC_SPEED_MM_S = 250  # мм/с — порог: меньше → статика, больше → динамика

# Подавление теней
SHADOW_MARGIN_DEG  = 30.0    # градусов — расширение углового конуса
SHADOW_DIST_RATIO  = 1.05   # тень дальше реального объекта минимум в N раз
SHADOW_ZONE_MM     = 3000   # мм — радиус зоны для правила 2
SHADOW_ZONE_DEG    = 20.0   # градусов — макс. угловое расстояние для правила 2

# Визуализация
IMG_W, IMG_H    = 1000, 1000
CENTER_X        = IMG_W // 2
CENTER_Y        = IMG_H // 2
SCALE_INIT      = 0.15
SCALE_MIN       = 0.01
SCALE_MAX       = 0.25
SCALE_STEP      = 1.25

# Цвета (BGR)
COL_GRID        = (40,  40,  40)
COL_GRID_TEXT   = (70,  70,  70)
COL_LIDAR       = (0,   200, 255)
COL_POINT_STAT  = (0,   220, 0  )
COL_POINT_DYN   = (0,   0,   255)
COL_BBOX        = (255, 255, 255)
COL_LABEL       = (255, 255, 255)
COL_ARROW       = (0,   165, 255)
COL_TRAIL_START = (40,  40,  180)
COL_TRAIL_END   = (180, 60,  255)
COL_STATS_BG    = (20,  20,  20 )


#  ФИЛЬТР КАЛМАНА

class KalmanFilter2D:
    def __init__(self, x0: float, y0: float, dt: float = FRAME_DT):
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        q_pos, q_vel = 50.0, 2000.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.diag([150.0, 150.0])
        self.x = np.array([[x0], [y0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([500.0, 500.0, 10000.0, 10000.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x: float, z_y: float):
        z = np.array([[z_x], [z_y]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self) -> tuple:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self) -> tuple:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def speed(self) -> float:
        vx, vy = self.vel
        return math.hypot(vx, vy)


#  ТРЕК

class Track:
    _next_id: int = 1

    def __init__(self, cx: float, cy: float, points: list):
        self.id         = Track._next_id
        Track._next_id += 1
        self.kf         = KalmanFilter2D(cx, cy)
        self.points     = points
        self.age        = 1
        self.lost       = 0
        self.is_dynamic = False
        self.trail      = deque(maxlen=TRAIL_LEN)
        self.trail.append((cx, cy))
        print(f"[+] Объект #{self.id} обнаружен  pos=({cx:.0f}, {cy:.0f}) мм  pts={len(points)}")

    def update_matched(self, cx: float, cy: float, points: list):
        self.kf.predict()
        self.kf.update(cx, cy)
        self.points = points
        self.lost   = 0
        self.age   += 1
        self.trail.append(self.kf.pos)
        if self.age >= MIN_AGE_DYNAMIC:
            was_dynamic     = self.is_dynamic
            self.is_dynamic = self.kf.speed > DYNAMIC_SPEED_MM_S
            if self.is_dynamic and not was_dynamic:
                print(f"[~] Объект #{self.id} ДВИЖЕТСЯ  v={self.kf.speed/1000:.2f} м/с")
            elif not self.is_dynamic and was_dynamic:
                print(f"[~] Объект #{self.id} остановился")

    def update_lost(self):
        self.kf.predict()
        self.lost += 1
        self.age  += 1

    @property
    def centroid(self) -> tuple:
        return self.kf.pos

    @property
    def dist_from_lidar(self) -> float:
        x, y = self.kf.pos
        return math.hypot(x, y)

    def __del__(self):
        try:
            print(f"[-] Объект #{self.id} потерян  age={self.age} кадров")
        except Exception:
            pass


#  КЛАСТЕРИЗАЦИЯ

_RAY_ANGLES = [math.radians(ANGLE_START + (ANGLE_RANGE / (NUM_POINTS - 1)) * i)
               for i in range(NUM_POINTS)]
_RAY_COS = [math.cos(a) for a in _RAY_ANGLES]
_RAY_SIN = [math.sin(a) for a in _RAY_ANGLES]


def cluster_scan(distances: list) -> list:
    xy = []
    for i, d in enumerate(distances):
        if DIST_MIN_MM <= d <= DIST_MAX_MM:
            xy.append((_RAY_COS[i] * d, _RAY_SIN[i] * d))
        else:
            xy.append(None)

    clusters, current = [], []
    prev_pt = None
    for pt in xy:
        if pt is None:
            if CLUSTER_MIN_PTS <= len(current) <= CLUSTER_MAX_PTS:
                clusters.append(current)
            current, prev_pt = [], None
            continue
        if prev_pt is not None:
            if math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1]) > CLUSTER_SPLIT_MM:
                if CLUSTER_MIN_PTS <= len(current) <= CLUSTER_MAX_PTS:
                    clusters.append(current)
                current = []
        current.append(pt)
        prev_pt = pt
    if CLUSTER_MIN_PTS <= len(current) <= CLUSTER_MAX_PTS:
        clusters.append(current)

    filtered = []
    for cl in clusters:
        xs = [p[0] for p in cl]
        ys = [p[1] for p in cl]
        w  = max(xs) - min(xs)
        h  = max(ys) - min(ys)
        max_side = max(w, h)
        min_side = min(w, h)

        # Фильтр минимального размера
        if max_side < CLUSTER_MIN_SIZE_MM:
            continue
        # Фильтр максимального размера — стены и углы комнаты
        if max_side > CLUSTER_MAX_SIZE_MM:
            continue
        # Фильтр соотношения сторон — стены вытянуты, препятствия компактны
        if min_side > 0 and (max_side / min_side) > CLUSTER_MAX_ASPECT:
            continue

        filtered.append(cl)
    return filtered


def centroid(cluster: list) -> tuple:
    n = len(cluster)
    return sum(p[0] for p in cluster) / n, sum(p[1] for p in cluster) / n


#  СОПОСТАВЛЕНИЕ

def match(clusters: list, tracks: list) -> tuple:
    if not clusters or not tracks:
        return [], list(range(len(clusters))), list(range(len(tracks)))

    c_pts = [centroid(cl) for cl in clusters]
    t_pts = [tr.kf.pos    for tr in tracks  ]

    D = np.array([[math.hypot(c_pts[i][0] - t_pts[j][0], c_pts[i][1] - t_pts[j][1])
                   for j in range(len(tracks))]
                  for i in range(len(clusters))])

    used_c, used_t, matches = set(), set(), []
    for dist, ci, ti in sorted([(D[i,j], i, j)
                                  for i in range(len(clusters))
                                  for j in range(len(tracks))]):
        if dist > MATCH_DIST_MM: break
        if ci in used_c or ti in used_t: continue
        matches.append((ci, ti))
        used_c.add(ci); used_t.add(ti)

    return (matches,
            [i for i in range(len(clusters)) if i not in used_c],
            [j for j in range(len(tracks))   if j not in used_t])


#  ПОДАВЛЕНИЕ ТЕНЕЙ

def suppress_shadows(tracks: list) -> None:
    """Подавление ложных динамических объектов: угловой конус + ближайший побеждает."""
    dynamic = [t for t in tracks if t.is_dynamic and t.lost == 0]
    if len(dynamic) < 2:
        return

    suppressed = set()

    def norm_angle(a, ref):
        while a - ref >  180: a -= 360
        while a - ref < -180: a += 360
        return a

    for t in dynamic:
        if t.id in suppressed:
            continue

        tx, ty   = t.centroid
        t_dist   = math.hypot(tx, ty)
        t_angle  = math.degrees(math.atan2(ty, tx))

        for real in dynamic:
            if real.id == t.id or real.id in suppressed:
                continue

            rx, ry    = real.centroid
            real_dist = math.hypot(rx, ry)

            if t_dist > real_dist * SHADOW_DIST_RATIO and real.points:
                angles = [math.degrees(math.atan2(p[1], p[0])) for p in real.points]
                a_min  = min(angles) - SHADOW_MARGIN_DEG
                a_max  = max(angles) + SHADOW_MARGIN_DEG
                t_norm = norm_angle(t_angle, (a_min + a_max) / 2)
                if a_min <= t_norm <= a_max:
                    suppressed.add(t.id)
                    break

            # Только если треки близко И в одном угловом направлении
            dist_between = math.hypot(tx - rx, ty - ry)
            if dist_between < SHADOW_ZONE_MM:
                # Угловое расстояние между треками с точки зрения лидара
                t_angle    = math.degrees(math.atan2(ty, tx))
                real_angle = math.degrees(math.atan2(ry, rx))
                ang_sep    = abs(norm_angle(t_angle - real_angle, 0))
                # Подавляем только если угловое расстояние мало — иначе разные объекты
                if ang_sep < SHADOW_ZONE_DEG:
                    if t_dist > real_dist:
                        suppressed.add(t.id)
                        break
                    else:
                        suppressed.add(real.id)

    for t in dynamic:
        if t.id in suppressed:
            t.is_dynamic = False


#  ВИЗУАЛИЗАЦИЯ

def to_px(x_mm: float, y_mm: float, scale: float) -> tuple:
    return (int(CENTER_X + x_mm * scale), int(CENTER_Y - y_mm * scale))


def draw(distances: list, tracks: list, scale: float,
         frame_idx: int, fps: float) -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # Сетка
    for r_m in range(2, int(DIST_MAX_MM / 1000) + 2, 2):
        r_px = int(r_m * 1000 * scale)
        if r_px > max(IMG_W, IMG_H): break
        cv2.circle(img, (CENTER_X, CENTER_Y), r_px, COL_GRID, 1)
        lx = CENTER_X + r_px + 4
        if lx < IMG_W - 20:
            cv2.putText(img, f"{r_m}m", (lx, CENTER_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_GRID_TEXT, 1)
    cv2.line(img, (CENTER_X, 0), (CENTER_X, IMG_H), COL_GRID, 1)
    cv2.line(img, (0, CENTER_Y), (IMG_W, CENTER_Y), COL_GRID, 1)

    # Все точки скана
    for i, d in enumerate(distances):
        if not (DIST_MIN_MM <= d <= DIST_MAX_MM): continue
        px, py = to_px(_RAY_COS[i] * d, _RAY_SIN[i] * d, scale)
        if 0 <= px < IMG_W and 0 <= py < IMG_H:
            img[py, px] = COL_POINT_STAT

    # Треки
    n_dynamic = 0
    for tr in tracks:
        if tr.lost > 0: continue
        cx_mm, cy_mm = tr.centroid
        cx_px, cy_px = to_px(cx_mm, cy_mm, scale)

        if tr.is_dynamic:
            n_dynamic += 1

            for x_mm, y_mm in tr.points:
                px, py = to_px(x_mm, y_mm, scale)
                if 0 <= px < IMG_W and 0 <= py < IMG_H:
                    img[py, px] = COL_POINT_DYN

            pxs = [to_px(x, y, scale) for x, y in tr.points]
            bx1 = min(p[0] for p in pxs) - 4
            bx2 = max(p[0] for p in pxs) + 4
            by1 = min(p[1] for p in pxs) - 4
            by2 = max(p[1] for p in pxs) + 4
            cv2.rectangle(img, (bx1, by1), (bx2, by2), COL_BBOX, 1)

            speed_ms = tr.kf.speed / 1000.0
            cv2.putText(img, f"#{tr.id}  {speed_ms:.2f} m/s", (bx1, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_LABEL, 1)

            vx, vy = tr.kf.vel
            ex_px, ey_px = to_px(cx_mm + vx * 1.5, cy_mm + vy * 1.5, scale)
            if math.hypot(ex_px - cx_px, ey_px - cy_px) > 5:
                cv2.arrowedLine(img, (cx_px, cy_px), (ex_px, ey_px),
                                COL_ARROW, 2, tipLength=0.25)

            trail = list(tr.trail)
            for k in range(1, len(trail)):
                p1 = to_px(*trail[k-1], scale)
                p2 = to_px(*trail[k],   scale)
                t  = k / len(trail)
                color = (
                    int(COL_TRAIL_START[0] + t*(COL_TRAIL_END[0]-COL_TRAIL_START[0])),
                    int(COL_TRAIL_START[1] + t*(COL_TRAIL_END[1]-COL_TRAIL_START[1])),
                    int(COL_TRAIL_START[2] + t*(COL_TRAIL_END[2]-COL_TRAIL_START[2])),
                )
                cv2.line(img, p1, p2, color, 1)

    cv2.circle(img, (CENTER_X, CENTER_Y), 6, COL_LIDAR, -1)

    stats = [f"Frame : {frame_idx}", f"FPS   : {fps:.1f}",
             f"Scale : {1.0/scale/1000*IMG_W/2:.1f} m",
             f"Tracks: {sum(1 for t in tracks if t.lost==0)}",
             f"Moving: {n_dynamic}"]
    sw, sh, pad = 150, 14*len(stats)+10, 6
    cv2.rectangle(img, (pad, pad), (pad+sw, pad+sh), COL_STATS_BG, -1)
    for i, line in enumerate(stats):
        cv2.putText(img, line, (pad+5, pad+14*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    cv2.putText(img, "Q-quit  R-reset  +/--zoom  SPACE-pause",
                (5, IMG_H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)
    return img


#  ПОДКЛЮЧЕНИЕ К HOKUYO

def decode_3char(s: str) -> int:
    if len(s) < 3: return 0
    return ((ord(s[0])-0x30) << 12) | ((ord(s[1])-0x30) << 6) | (ord(s[2])-0x30)

def hokuyo_send(sock, cmd: str):
    sock.sendall((cmd + "\n").encode("ascii"))

def hokuyo_recv(sock) -> str:
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk: break
        data += chunk
        if b"\n\n" in data: break
    return data.decode("ascii")

def parse_response(response: str) -> list | None:
    lines    = response.split("\n")
    raw_data = "".join(line[:-1] for line in lines if len(line) > 3)
    if len(raw_data) < 3: return None
    distances = [decode_3char(raw_data[i:i+3])/80
                 for i in range(0, len(raw_data)-2, 3)]
    distances = distances[11:]
    if len(distances) != NUM_POINTS:
        print(f"[Hokuyo] Дефектный кадр: {len(distances)} точек — пропуск")
        return None
    return distances


#  ГЛАВНЫЙ ЦИКЛ

def main():
    print("=" * 60)
    print("  Детекция динамических препятствий — Hokuyo UTM-30LX")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {LIDAR_IP}:{LIDAR_PORT}")
    print("=" * 60)

    scale      = SCALE_INIT
    tracks     = []
    frame_idx  = 0
    paused     = False
    fps        = 0.0
    t_fps      = time.perf_counter()
    fps_frames = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect((LIDAR_IP, LIDAR_PORT))
        print("✓ Подключено к лидару")

        hokuyo_send(sock, "BM")
        time.sleep(0.1)

        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if   key == ord("q"): break
                elif key == ord(" "): paused = False
                continue

            hokuyo_send(sock, "MD0000108000001")
            time.sleep(0.05)
            response  = hokuyo_recv(sock)
            distances = parse_response(response)
            if distances is None: continue

            frame_idx  += 1
            fps_frames += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                fps        = fps_frames / (now - t_fps)
                fps_frames = 0
                t_fps      = now

            clusters = cluster_scan(distances)
            matches, new_c, lost_t = match(clusters, tracks)

            for ci, ti in matches:
                cx, cy = centroid(clusters[ci])
                tracks[ti].update_matched(cx, cy, clusters[ci])
            for ti in lost_t:
                tracks[ti].update_lost()
            for ci in new_c:
                cx, cy = centroid(clusters[ci])
                tracks.append(Track(cx, cy, clusters[ci]))

            tracks = [t for t in tracks if t.lost <= MAX_LOST_FRAMES]
            suppress_shadows(tracks)

            img = draw(distances, tracks, scale, frame_idx, fps)
            cv2.imshow("Dynamic Obstacle Detection", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord(" "):
                paused = True; print("[Пауза]")
            elif key == ord("r"):
                tracks.clear(); Track._next_id = 1; print("[Треки сброшены]")
            elif key in (ord("+"), ord("=")):
                scale = min(SCALE_MAX, scale * SCALE_STEP)
            elif key == ord("-"):
                scale = max(SCALE_MIN, scale / SCALE_STEP)

        hokuyo_send(sock, "QT")
        hokuyo_recv(sock)
        cv2.destroyAllWindows()
        print(f"\nЗавершено. Обработано кадров: {frame_idx}")


if __name__ == "__main__":
    main()
