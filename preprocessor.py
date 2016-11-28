import cv2


class PreProcessor:
    def __init__(self, pretrain_dataset):
        self.face_cascade = cv2.CascadeClassifier(pretrain_dataset)

    '''
    Convert image to black scale
    detect face in image
    return a array of face size 96x96
    '''
    def process_file(self, filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret = []
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            ret.push(roi_gray)
        return ret


