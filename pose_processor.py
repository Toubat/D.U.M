import cv2
import mediapipe as mp
import numpy as np

class PoseDetector():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.discard_points = (1, 2, 3, 4, 5, 6, 9, 10, 21, 22)

    def get_video_keypoints(self, path='./sample.mp4'):
        cap = cv2.VideoCapture(path)
        keypoints = []
        originals = []
        while cap.isOpened():
            _, img = cap.read()
            if (img is None):
                break
            keypoint = []
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.pose.process(img_RGB)
            if result.pose_landmarks:
                # self.mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(result.pose_landmarks.landmark):
                    if id in self.discard_points:
                        continue
                    height, width, channel = img.shape
                    # cx, cy = int(lm.x * width), int(lm.y * height)
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    # normalize pkeypoints (value range between -1 and 1)
                    keypoint.append(lm.x * 2 - 1)
                    keypoint.append(lm.y * 2 - 1)
                    keypoint.append(lm.z)
                keypoints.append(keypoint)   
                originals.append(result.pose_landmarks)
            else:
                keypoints.append(keypoints[-1])  
                originals.append(originals[-1])
            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # cv2.imshow("Image", img)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        # [600, 69]
        keypoints = np.array(keypoints) # convert into numpy array

        return keypoints, originals

"""
/video
    gLO_sBM_c03_d15_mLO5_ch04.npy
    gLO_sBM_c03_d15_mLO5_ch04.npy
    gLO_sBM_c03_d15_mLO5_ch04.npy
    gLO_sBM_c03_d15_mLO5_ch04.npy

"""


def video_show(originals, path='./sample.mp4'):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(path)
    i = 0

    while(cap.isOpened()):
        _, img = cap.read()    
        if (img is None):
                break
        mp_draw.draw_landmarks(img, originals[i], mp_pose.POSE_CONNECTIONS)  
        cv2.imshow('frame',img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    path = 'path/to/video.mp4'
    detector = PoseDetector()
    keypoints, _ = detector.get_video_keypoints('https://s3.abci.ai/aistdancedb.ongaaccel.jp/v1.0.0/video/10M/gBR_sBM_c07_d04_mBR1_ch08.mp4')
    print(keypoints.shape)

    return 0

if __name__ == '__main__':
    main()