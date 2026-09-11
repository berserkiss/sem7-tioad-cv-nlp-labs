import cv2
import argparse
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=0, help='Путь к видео или индекс камеры')
    p.add_argument('--min_area', type=int, default=15000, help='Минимальная площадь контура')
    p.add_argument('--min_w', type=int, default=40, help='Минимальная ширина контура')
    p.add_argument('--min_h', type=int, default=40, help='Минимальная высота контура')
    p.add_argument('--scale', type=float, default=0.5, help='Коэффициент предмасштабирования ROI')
    return p.parse_args()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)

    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить первый кадр")
        return
    orig_h, orig_w = frame.shape[:2]
    cv2.namedWindow('Результат', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Результат', orig_w, orig_h)
    cv2.namedWindow('Маска', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Маска', orig_w, orig_h)

    # Выбор ROI
    small = cv2.resize(frame, (int(orig_w * args.scale), int(orig_h * args.scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Выбор ROI", small)
    cv2.waitKey(1)
    x_s, y_s, w_s, h_s = cv2.selectROI("Выбор ROI", small, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow("Выбор ROI")
    x, y = int(x_s / args.scale), int(y_s / args.scale)
    w_roi, h_roi = int(w_s / args.scale), int(h_s / args.scale)

    # Инициализация трекера
    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker.init(frame, (x, y, w_roi, h_roi))

    # Фоновое вычитание
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # морфология для очистки
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mask_flow = np.zeros_like(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Детектирование движения по фону
        fg = fgbg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion = False
        for c in cnts:
            area = cv2.contourArea(c)
            x_c, y_c, w_c, h_c = cv2.boundingRect(c)
            if area < args.min_area or w_c < args.min_w or h_c < args.min_h:
                continue
            cv2.rectangle(frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
            motion = True

        if motion:
            print("Motion detected!\a")  # вывод сообщения и звуковой сигнал

        # 2) Оптический поток Лукаса–Канаде
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # Сброс маски потока на каждом кадре (исправление для удаления накопленных линий)
            mask_flow = np.zeros_like(frame)
            for new, old in zip(good_new, good_old):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())
                mask_flow = cv2.line(mask_flow, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
            frame = cv2.add(frame, mask_flow)
            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        # 3) Трекинг MOSSE
        ok, bb = tracker.update(frame)
        if ok:
            x_t, y_t, w_t, h_t = map(int, bb)
            cv2.rectangle(frame, (x_t, y_t), (x_t + w_t, y_t + h_t), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Показ результатов
        cv2.imshow('Результат', frame)
        cv2.imshow('Маска', fg)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




# import argparse
# import numpy as np
#
#
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--source', default=0, help='Путь к видео или индекс камеры')
#     p.add_argument('--min_area', type=int, default=15000, help='Минимальная площадь контура')
#     p.add_argument('--min_w', type=int, default=40, help='Минимальная ширина контура')
#     p.add_argument('--min_h', type=int, default=40, help='Минимальная высота контура')
#     p.add_argument('--scale', type=float, default=0.5, help='Коэффициент предмасштабирования ROI')
#     return p.parse_args()
#
#
# def main():
#     args = parse_args()
#     cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
#
#     ret, frame = cap.read()
#     if not ret:
#         print("Не удалось получить первый кадр")
#         return
#     orig_h, orig_w = frame.shape[:2]
#     cv2.namedWindow('Результат', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Результат', orig_w, orig_h)
#     cv2.namedWindow('Маска', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Маска', orig_w, orig_h)
#
#     # Выбор ROI
#     small = cv2.resize(frame, (int(orig_w * args.scale), int(orig_h * args.scale)), interpolation=cv2.INTER_AREA)
#     cv2.imshow("Выбор ROI", small)
#     cv2.waitKey(1)
#     x_s, y_s, w_s, h_s = cv2.selectROI("Выбор ROI", small, showCrosshair=False, fromCenter=False)
#     cv2.destroyWindow("Выбор ROI")
#     x, y = int(x_s / args.scale), int(y_s / args.scale)
#     w_roi, h_roi = int(w_s / args.scale), int(h_s / args.scale)
#
#     # Инициализация трекера
#     tracker = cv2.legacy.TrackerMOSSE_create()
#     tracker.init(frame, (x, y, w_roi, h_roi))
#
#     # Фоновое вычитание
#     fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # морфология для очистки
#     old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
#     lk_params = dict(winSize=(15, 15), maxLevel=2,
#                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     mask_flow = np.zeros_like(frame)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 1) Детектирование движения по фону
#         fg = fgbg.apply(frame)
#         fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
#         fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
#         cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         motion = False
#         for c in cnts:
#             area = cv2.contourArea(c)
#             x_c, y_c, w_c, h_c = cv2.boundingRect(c)
#             if area < args.min_area or w_c < args.min_w or h_c < args.min_h:
#                 continue
#             cv2.rectangle(frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
#             motion = True
#
#         if motion:
#             print("Motion detected!\a")  # вывод сообщения и звуковой сигнал
#
#         # 2) Оптический поток Лукаса–Канаде
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
#         if p1 is not None:
#             good_new = p1[st == 1]
#             good_old = p0[st == 1]
#             for new, old in zip(good_new, good_old):
#                 a, b = map(int, new.ravel())
#                 c, d = map(int, old.ravel())
#                 mask_flow = cv2.line(mask_flow, (a, b), (c, d), (0, 255, 0), 2)
#                 cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
#             frame = cv2.add(frame, mask_flow)
#             old_gray = gray.copy()
#             p0 = good_new.reshape(-1, 1, 2)
#
#         # 3) Трекинг MOSSE
#         ok, bb = tracker.update(frame)
#         if ok:
#             x_t, y_t, w_t, h_t = map(int, bb)
#             cv2.rectangle(frame, (x_t, y_t), (x_t + w_t, y_t + h_t), (255, 0, 0), 2)
#         else:
#             cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # Показ результатов
#         cv2.imshow('Результат', frame)
#         cv2.imshow('Маска', fg)
#         if cv2.waitKey(30) == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()