import cv2
def list_camera_devices():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

if __name__ == "__main__":
    available_cameras = list_camera_devices()
    print(f"Available Camera Devices: {available_cameras}")