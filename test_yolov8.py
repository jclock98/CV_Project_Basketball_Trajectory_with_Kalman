from ultralytics import YOLO

from kalmanfilter import KalmanFilter
from utils import *

MAX_FRAMES = 1000000
CUT_FACTOR = 7
REDU = 8
THRESHOLD = 60


def init_tracker():
    return YOLO("utils/best_s.pt")


def init_kalman():
    kernel = np.ones((3, 3), np.uint8)

    fps = 120
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
    N = 15
    sigmaM = 0.0001
    sigmaZ = 3 * noise

    Q = sigmaM ** 2 * np.eye(4)
    R = sigmaZ ** 2 * np.eye(2)

    return kernel, A, B, u, H, mu, P, res, N, Q, R


def track_trajectory(input_file):
    bgsub = cv2.createBackgroundSubtractorMOG2(500, 60, True)

    model = init_tracker()

    found = False
    add_count = 0
    clean = False
    fps = 120

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    kf = KalmanFilter()

    kernel, A, B, u, H, mu, P, res, N, Q, R = init_kalman()

    listCenterX = []
    listCenterY = []

    cap = cv2.VideoCapture(input_file)

    while True:  # for t in range(MAX_FRAMES):
        # Capture frame-by-frame
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            break

        ret, frame = cap.read()
        # If video end reached
        if not ret:
            break
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
            cv2.imshow('Frame', frame)
            if results[0].boxes.shape[0] == 0:
                continue
            else:
                found = True
                print("Ball found...")
                bbox = results[0].boxes.xyxy.cpu().numpy()[0]
                x_t_l, y_t_l, x_b_r, y_b_r = list(map(lambda x: int(x), bbox))
                x_b_r -= CUT_FACTOR
                y_b_r -= CUT_FACTOR
                x_t_l += CUT_FACTOR
                y_t_l += CUT_FACTOR
                imCrop = frame[y_t_l:y_b_r, x_t_l:x_b_r]
                print("Cropping ball...")
                imCropMask = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
                ret, imCropMask = cv2.threshold(imCropMask, 30, 255, cv2.THRESH_BINARY)
                his = smooth(1, rgbh([imCrop], imCropMask))
                roiBox = (x_t_l, y_t_l, x_b_r - x_t_l, y_b_r - y_t_l)
                print("Done")
        else:
            print("Starting to track the ball...")
            add_count += 1
            rgbr = np.floor_divide(colorMask, REDU)
            r, g, b = rgbr.transpose(2, 0, 1)
            l = his[r, g, b]

            print("Track ball...")
            (rb, roiBox) = cv2.CamShift(l, roiBox, termination)

            cv2.ellipse(frame, rb, (0, 255, 0), 2)
            xo = int(roiBox[0] + roiBox[2] / 2)
            yo = int(roiBox[1] + roiBox[3] / 2)
            error = (roiBox[3])

            print("Predicting trajectory...")
            if yo < error or bgs.sum() < 50:
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
                               5,
                               (255, 255, 0),
                               3)

                for n in range(len(xp)):  # x, y prediction
                    uncertaintyP = (xpu[n] + ypu[n]) / 2
                    cv2.circle(frame,
                               (int(xp[n]), int(yp[n])),
                               int(uncertaintyP),
                               (0, 0, 255))

            if len(listCenterY) > 4:
                check_rebound_y_towards_left = ((listCenterY[-4] > listCenterY[-3]) and
                                                (listCenterY[-3] > listCenterY[-2]) and
                                                (listCenterY[-2] < listCenterY[-1]))
                check_rebound_x_towards_left = ((listCenterX[-4] > listCenterX[-3]) and
                                                (listCenterX[-3] > listCenterX[-2]) and
                                                (listCenterX[-2] < listCenterX[-1]))

                check_rebound_y_towards_right = ((listCenterY[-4] < listCenterY[-3]) and
                                                 (listCenterY[-3] < listCenterY[-2]) and
                                                 (listCenterY[-2] > listCenterY[-1]))
                check_rebound_x_towards_right = ((listCenterX[-4] < listCenterX[-3]) and
                                                 (listCenterX[-3] < listCenterX[-2]) and
                                                 (listCenterX[-2] > listCenterX[-1]))

                # check for rebound in either direction
                if (check_rebound_x_towards_right or check_rebound_x_towards_left) and\
                        (check_rebound_y_towards_right or check_rebound_y_towards_left):
                    clean = True

            if clean:
                print("Reset")
                listCenterY = []
                listCenterX = []
                res = []
                clean = False
                found = False
                mu = np.array([0, 0, 0, 0])
                P = np.diag([10, 10, 10, 10]) ** 2

        cv2.imshow('Frame', frame)

    print('done')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = f"data/01.mp4"

    track_trajectory(filename)
