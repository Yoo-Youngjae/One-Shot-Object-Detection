import socket
import cv2
import numpy
import qi
import argparse
import sys
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt

def send_image(im):
    TCP_IP = '147.46.219.160'
    TCP_PORT = 5001


    frame = np.array(im)


    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    decimg = cv2.imdecode(data, 1)
    plt.imshow(decimg)
    plt.show()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))


    sock.send(str(len(stringData)).ljust(16))
    sock.send(stringData)
    sock.close()


    # decimg=cv2.imdecode(data,1)
    # cv2.imshow('CLIENT',decimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def shot_rgb_t_camera(session):
    video_service = session.service("ALVideoDevice")
    fps = 20
    resolution = 2      # 0 == 160, 120 | 1 == 320, 240 | 2 == 640, 480
    colorSpace = 11     # 11 == bgr?
    name_id = video_service.subscribe("rgb_b", resolution, colorSpace, fps)
    # video_service.subscribe(name, idx, resolution, color space, fps)

    for i in range(1):
        pepper_img = video_service.getImageRemote(name_id)
        width, height = pepper_img[0], pepper_img[1]
        array = pepper_img[6]
        img_str = str(bytearray(array))
        im = Image.frombytes("RGB", (width, height), img_str)
        send_image(im)

    video_service.unsubscribe(name_id)

### config ####
PEPPER_IP = '192.168.1.123'
parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default=PEPPER_IP,
                    help="Robot IP address. On robot or Local Naoqi: use '192.168.1.123'.")
parser.add_argument("--port", type=int, default=9559,
                    help="Naoqi port number")
args = parser.parse_args()

##################################
if __name__ == "__main__":

    session = qi.Session()
    try:
        session.connect("tcp://" + PEPPER_IP + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                          "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    # shot_camera(session)
    shot_rgb_t_camera(session)