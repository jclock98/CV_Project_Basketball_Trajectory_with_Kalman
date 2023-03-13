import argparse
from typing import List, Dict

from numpy import ndarray
from ultralytics import YOLO

from kalmanfilter import KalmanFilter
from utils import *

MAX_FRAMES = 1000000
CUT_FACTOR = 8
REDU = 8
THRESHOLD = 60
FRAMES_INTERVAL = 30


def init_tracker(model_type: str) -> YOLO:
    load_model = YOLO(f"models/best_{model_type}.pt")
    load_model.to("cpu")
    return load_model


def init_kalman(fps: int) -> (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, List, ndarray, ndarray):

    kernel = np.ones((3, 3), np.uint8)

    dt = 1 / fps
    noise = 3

    A = np.array(
        [1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1]).reshape(4, 4)

    # gravity correction
    u = np.array([0, 5])
    B = np.array(
        [dt ** 2 / 2, 0,
         0, dt ** 2 / 2,
         dt, 0,
         0, dt]).reshape(4, 2)

    H = np.array(
        [1, 0, 0, 0,
         0, 1, 0, 0]).reshape(2, 4)

    # x, y, vx, vy
    mu = np.array([0, 0, 0, 0])
    P = np.diag([10, 10, 10, 10]) ** 2

    res = []
    sigmaM = 0.0001
    sigmaZ = 3 * noise

    Q = sigmaM ** 2 * np.eye(4)
    R = sigmaZ ** 2 * np.eye(2)

    return kernel, A, B, u, H, mu, P, res, Q, R


def get_bbox_coords(results: Dict) -> (int, int, int, int):
    bbox = results[0].boxes.xyxy.numpy()[0]
    x_t_l, y_t_l, x_b_r, y_b_r = list(map(lambda x: int(x), bbox))
    x_b_r -= CUT_FACTOR
    y_b_r -= CUT_FACTOR
    x_t_l += CUT_FACTOR
    y_t_l += CUT_FACTOR
    return x_t_l, y_t_l, x_b_r, y_b_r


def track_trajectory(input_file: str, type_of_model: str, camshift: bool, write: bool, show: bool):
    bgsub = cv2.createBackgroundSubtractorMOG2(500, 60, True)
    model = init_tracker(type_of_model)

    found = False
    clean = False
    fps = 120

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    kf = KalmanFilter()

    kernel, A, B, u, H, mu, P, res, Q, R = init_kalman(fps)

    # lists of boxes' centre
    listCenterX = []
    listCenterY = []
    frames = []

    cap = cv2.VideoCapture(input_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_num = 0

    while True:
        # Capture frame-by-frame
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break

        # read frame
        ret, frame = cap.read()
        frame_num += 1
        # If video end reached
        if not ret:
            break

        # compute color mask
        bgs = bgsub.apply(frame)
        bgs = cv2.erode(bgs, kernel, iterations=1)
        bgs = cv2.medianBlur(bgs, 3)
        bgs = cv2.dilate(bgs, kernel, iterations=2)
        bgs = (bgs > 200).astype(np.uint8) * 255
        colorMask = cv2.bitwise_and(frame, frame, mask=bgs)

        # If basketball has not been found yet
        if not found:
            print("Detecting ball...")
            results = model(frame)
            if show:
                cv2.imshow('Frame', frame)
            frames.append(frame)
            if results[0].boxes.shape[0] == 0:
                continue
            else:
                found = True
                print("Ball found...")
                x_t_l, y_t_l, x_b_r, y_b_r = get_bbox_coords(results)
                selection = frame[y_t_l:y_b_r, x_t_l:x_b_r]
                print("Cropping ball...")
                if camshift:
                    selection_mask = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
                    ret, selection_mask = cv2.threshold(selection_mask, 30, 255, cv2.THRESH_BINARY)
                    his = smooth(1, rgbh([selection], selection_mask))
                selection_box = (x_t_l, y_t_l, x_b_r - x_t_l, y_b_r - y_t_l)
                print("Done")
        else:
            print("Track ball...")
            if camshift:
                rgbr = np.floor_divide(colorMask, REDU)
                r, g, b = rgbr.transpose(2, 0, 1)
                l = his[r, g, b]
                (rb, selection_box) = cv2.CamShift(l, selection_box, termination)
                cv2.ellipse(frame, rb, (0, 255, 0), 2)
                error = (selection_box[3])
                conf = 1
            else:
                results = model(frame)
                if results[0].boxes.shape[0] != 0:
                    print("Ball found...")
                    x_t_l, y_t_l, x_b_r, y_b_r = get_bbox_coords(results)
                    selection_box = (x_t_l, y_t_l, x_b_r - x_t_l, y_b_r - y_t_l)
                    conf = results[0].boxes[0].cpu().conf
                    error = 0
                else:
                    conf = 1
                    error = 0

                cv2.rectangle(frame, (x_t_l, y_t_l), (x_b_r, y_b_r), (0, 255, 0), 2)

            xo = int(selection_box[0] + selection_box[2] / 2)
            yo = int(selection_box[1] + selection_box[3] / 2)

            print("Predicting trajectory...")
            if bgs.sum() < 25 or conf < 0.05 or yo < error:
                predicted, mu, statePost, errorCovPre = kf.predict(int(xo), int(yo))
                mu, P = kf.kal(mu, P, B, u, z=None)
                mm = False
            else:
                predicted, mu, statePost, errorCovPre = kf.predict(int(xo), int(yo))
                mu, P = kf.kal(mu, P, B, u, z=np.array([xo, yo]))
                mm = True

            if mm:
                listCenterX.append(xo)
                listCenterY.append(yo)

            if len(listCenterX) > 2:
                res += [(mu, P)]
                cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 255), 3)

                # Prediction #
                mu2 = mu
                P2 = P
                res2 = []

                for _ in range(fps * 2):
                    mu2, P2 = kf.kal(mu2, P2, B, u, z=None)
                    res2 += [(mu2, P2)]

                xe = [mu[0] for mu, _ in res]
                xu = [2 * np.sqrt(P[0, 0]) for _, P in res]
                ye = [mu[1] for mu, _ in res]
                yu = [2 * np.sqrt(P[1, 1]) for _, P in res]

                xp = [mu2[0] for mu2, _ in res2]  # first res2 is not used
                yp = [mu2[1] for mu2, _ in res2]

                xpu = [np.sqrt(P[0, 0]) for _, P in res2]
                ypu = [np.sqrt(P[1, 1]) for _, P in res2]

                # ball trace
                for n in range(len(listCenterX)):  # centre of prediction
                    cv2.circle(frame,
                               (int(listCenterX[n]), int(listCenterY[n])),
                               3,
                               (0, 255, 0),
                               -1)

                # predict location
                for n in [-1]:
                    uncertainty = (xu[n] + yu[n]) / 2
                    cv2.circle(frame,
                               (int(xe[n]), int(ye[n])),
                               int(uncertainty),
                               (255, 255, 0),
                               3)

                for n in range(len(xp)):  # x, y prediction
                    uncertaintyP = (xpu[n] + ypu[n]) / 2
                    cv2.circle(frame,
                               (int(xp[n]), int(yp[n])),
                               int(uncertaintyP),
                               (0, 0, 255))

            if len(listCenterY) > 3:
                check_rebound_y_towards_left = (  # (listCenterY[-4] > listCenterY[-3]) and
                                                (listCenterY[-3] > listCenterY[-2]) and
                                                (listCenterY[-2] < listCenterY[-1]))
                check_rebound_x_towards_left = (  # (listCenterX[-4] > listCenterX[-3]) and
                                                (listCenterX[-3] > listCenterX[-2]) and
                                                (listCenterX[-2] < listCenterX[-1]))

                check_rebound_y_towards_right = (  # (listCenterY[-4] < listCenterY[-3]) and
                                                 (listCenterY[-3] < listCenterY[-2]) and
                                                 (listCenterY[-2] > listCenterY[-1]))
                check_rebound_x_towards_right = (  # (listCenterX[-4] < listCenterX[-3]) and
                                                 (listCenterX[-3] < listCenterX[-2]) and
                                                 (listCenterX[-2] > listCenterX[-1]))
                print(listCenterX[-3:], listCenterY[-3:])
                # check for rebound in either direction
                if (check_rebound_x_towards_right or check_rebound_x_towards_left) and\
                        (check_rebound_y_towards_right or check_rebound_y_towards_left):
                    clean = True
            print(clean)
            if clean:
                print("Reset")
                listCenterY = []
                listCenterX = []
                res = []
                clean = False
                found = False
                mu = np.array([0, 0, 0, 0])
                P = np.diag([10, 10, 10, 10]) ** 2
            if camshift:
                if frame_num % FRAMES_INTERVAL == 0:
                    found = False

        if show:
            cv2.imshow('Frame', frame)
        frames.append(frame)

    if write:
        print("Writing video...")
        name = input_file[:-4] + "_tracked" + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, FRAMES_INTERVAL//2, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()
    print("Done")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for basketball shot trajectory prediction "
                                                 "using YOLO and Kalman Filter")
    parser.add_argument("--input", type=str, default="01.mp4",
                        help="Name of the input video (must be in data folder)")
    parser.add_argument("--model", type=str, default="small",
                        help="Type of model (nano, small, medium)")
    parser.add_argument("--camshift", default=False, action="store_true",
                        help="Whether to track the ball with camshift or YOLO")
    parser.add_argument("--save-result", default=True, action="store_true",
                        help="Save the resulting video with the predicted trajectory")
    parser.add_argument("--show", default=True, action="store_true",
                        help="Show procedure step-by-step")
    args = parser.parse_args()

    filename = f"data/{args.input}"
    model = args.model[0]
    camshift_tracker = args.camshift
    save_video = args.save_result
    show = args.show

    track_trajectory(filename, model, camshift_tracker, save_video, show)
