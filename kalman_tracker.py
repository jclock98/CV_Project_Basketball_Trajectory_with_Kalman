import argparse

from ultralytics import YOLO

from const import REDU, FRAMES_INTERVAL, THRESHOLD, CONF_THRESHOLD, FPS
from kalmanfilter import KalmanFilter
from utils import *


class KalmanTracker:

    def __init__(self, model_type: str, device: str):
        # -----INITIALIZE MODEL-----
        self.model = YOLO(f"models/best_{model_type}.pt")
        self.model.to(device)

        # -----INITIALIZE KALMAN FILTER-----
        self.kernel = np.ones((3, 3), np.uint8)

        dt = 1 / FPS
        noise = 3

        self.A = np.array(
            [1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1]).reshape(4, 4)

        # gravity correction
        self.u = np.array([0, 5])
        self.B = np.array(
            [dt ** 2 / 2, 0,
             0, dt ** 2 / 2,
             dt, 0,
             0, dt]).reshape(4, 2)

        self.H = np.array(
            [1, 0, 0, 0,
             0, 1, 0, 0]).reshape(2, 4)

        self.mu = np.array([0, 0, 0, 0])
        self.P = np.diag([10, 10, 10, 10]) ** 2

        self.res = []
        sigmaM = 0.0001
        sigmaZ = 3 * noise

        self.Q = sigmaM ** 2 * np.eye(4)
        self.R = sigmaZ ** 2 * np.eye(2)

        # -----INITIALIZE TRACKER PROPERTIES-----
        self.bgsub = cv2.createBackgroundSubtractorMOG2(500, 60, True)

        # lists of boxes' centre
        self.list_centre_x = []
        self.list_centre_y = []

        self.frames = []
        self.found = False

    def write_video(self, input_file: str, frame_width: int, frame_height: int, camshift: bool, model_type: str):
        print("Writing video...")
        camshift_label = "_camshift_" if camshift else ""
        name = input_file[:-4] + "_tracked" + f"_{model_type}_" + camshift_label + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, FRAMES_INTERVAL // 2, (frame_width, frame_height))
        for frame in self.frames:
            out.write(frame)
        out.release()


def track_trajectory(input_file: str, type_of_model: str, camshift: bool, write: bool, show_res: bool):
    clean = False
    tracker = KalmanTracker(type_of_model, device="cpu")

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    kf = KalmanFilter()

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

        # If basketball has not been found yet
        if not tracker.found:
            print("Detecting ball...")
            results = tracker.model(frame)

            tracker.frames.append(frame)

            if show_res:
                cv2.imshow('Frame', frame)

            if results[0].boxes.shape[0] == 0:
                continue
            else:
                tracker.found = True

                print("Ball found...")
                x_top_l, y_top_l, x_bot_r, y_bot_r = get_bbox_coords(results)
                selection = frame[y_top_l:y_bot_r, x_top_l:x_bot_r]

                print("Cropping ball...")
                if camshift:
                    selection_mask = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
                    ret, selection_mask = cv2.threshold(selection_mask, 30, 255, cv2.THRESH_BINARY)
                    hist = smooth(1, rgbh([selection], selection_mask))
                selection_box = (x_top_l, y_top_l, x_bot_r - x_top_l, y_bot_r - y_top_l)
                print("Done")
        else:

            print("Track ball...")
            if camshift:

                # compute color mask
                bgs = tracker.bgsub.apply(frame)
                bgs = cv2.erode(bgs, tracker.kernel, iterations=1)
                bgs = cv2.medianBlur(bgs, 3)
                bgs = cv2.dilate(bgs, tracker.kernel, iterations=2)
                bgs = (bgs > 200).astype(np.uint8) * 255
                colorMask = cv2.bitwise_and(frame, frame, mask=bgs)
                rgbr = np.floor_divide(colorMask, REDU)
                r, g, b = rgbr.transpose(2, 0, 1)
                loc = hist[r, g, b]

                (rb, selection_box) = cv2.CamShift(loc, selection_box, termination)
                # print selection box
                cv2.ellipse(frame, rb, (0, 255, 0), 2)
                error = (selection_box[3])
                conf = 1
            else:
                results = tracker.model(frame)
                if results[0].boxes.shape[0] != 0:
                    print("Ball found...")
                    x_top_l, y_top_l, x_bot_r, y_bot_r = get_bbox_coords(results)
                    selection_box = (x_top_l, y_top_l, x_bot_r - x_top_l, y_bot_r - y_top_l)
                    conf = results[0].boxes[0].cpu().conf
                    error = 0
                else:
                    conf = 1
                    error = 0
                # print selection box
                cv2.rectangle(frame, (x_top_l, y_top_l), (x_bot_r, y_bot_r), (0, 255, 0), 2)

            xo = int(selection_box[0] + selection_box[2] / 2)
            yo = int(selection_box[1] + selection_box[3] / 2)

            print("Predicting trajectory...")
            # no ball/movement detected
            # i.e. no white in the difference between frames, low confidence on detection or box under error threshold
            if (camshift and bgs.sum() < 25) or conf < CONF_THRESHOLD or yo < error:
                predicted, tracker.mu, _ = kf.predict(int(xo), int(yo))
                tracker.mu, tracker.P = kf.kal(tracker.mu, tracker.P, tracker.B, tracker.u, z=None)
            else:
                predicted, tracker.mu, _ = kf.predict(int(xo), int(yo))
                tracker.mu, tracker.P = kf.kal(tracker.mu, tracker.P, tracker.B, tracker.u, z=np.array([xo, yo]))
                tracker.list_centre_x.append(xo)
                tracker.list_centre_y.append(yo)

            if len(tracker.list_centre_x) > 2:
                # make prediction for trajectory
                tracker.res += [(tracker.mu, tracker.P)]
                cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 255), 3)

                # Prediction #
                mu2 = tracker.mu
                P2 = tracker.P
                res2 = []

                for _ in range(FPS * 2):
                    mu2, P2 = kf.kal(mu2, P2, tracker.B, tracker.u, z=None)
                    res2 += [(mu2, P2)]

                xe = [mu[0] for mu, _ in tracker.res]
                xu = [2 * np.sqrt(P[0, 0]) for _, P in tracker.res]
                ye = [mu[1] for mu, _ in tracker.res]
                yu = [2 * np.sqrt(P[1, 1]) for _, P in tracker.res]

                xp = [mu2[0] for mu2, _ in res2]  # first res2 is not used
                yp = [mu2[1] for mu2, _ in res2]

                xpu = [np.sqrt(P[0, 0]) for _, P in res2]
                ypu = [np.sqrt(P[1, 1]) for _, P in res2]

                # ball trace
                for n in range(len(tracker.list_centre_x)):  # centre of prediction
                    cv2.circle(frame,
                               (int(tracker.list_centre_x[n]), int(tracker.list_centre_y[n])),
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

            if len(tracker.list_centre_y) > 3:
                check_rebound_y_upward_bounce = ((tracker.list_centre_y[-3] <= tracker.list_centre_y[-2]) and
                                                 (tracker.list_centre_y[-2] >= tracker.list_centre_y[-1]))

                check_x_dist_between_meas = (
                            (abs(tracker.list_centre_x[-3] - tracker.list_centre_x[-2]) > THRESHOLD) and
                            (abs(tracker.list_centre_x[-2] - tracker.list_centre_x[-1]) > THRESHOLD))

                # check for rebound in either direction
                if check_rebound_y_upward_bounce and check_x_dist_between_meas:
                    clean = True

            if camshift:
                if frame_num % FRAMES_INTERVAL == 0:
                    tracker.found = False

            if clean:
                print("Reset")
                tracker.list_centre_y = []
                tracker.list_centre_x = []
                tracker.res = []
                clean = False
                tracker.found = False
                tracker.mu = np.array([0, 0, 0, 0])
                tracker.P = np.diag([10, 10, 10, 10]) ** 2

        if show_res:
            cv2.imshow('Frame', frame)
        tracker.frames.append(frame)

    if write:
        tracker.write_video(input_file, frame_width, frame_height, camshift, type_of_model)

    print("Done")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for basketball shot trajectory prediction "
                                                 "using YOLO and Kalman Filter")
    parser.add_argument("--input", type=str, default="01.mp4",
                        help="Name of the input video (must be in data folder)")
    parser.add_argument("--model", type=str, default="medium",
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
