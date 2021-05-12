import socket
import cv2
import numpy
import qi
import argparse
import sys
from PIL import Image
import numpy as np
import pandas as pd
import time
from datetime import datetime
import almath
from matplotlib import pyplot as plt
from matplotlib import image as img

def send_image(im, sock):

    frame = np.array(im)
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    sock.send(str(len(stringData)).ljust(16))
    sock.send(stringData)
    # decimg = cv2.imdecode(data, 1)
    # plt.imshow(decimg)
    # plt.show()

    # decimg=cv2.imdecode(data,1)
    # cv2.imshow('CLIENT',decimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def shot_rgb_t_camera(session):
    video_service = session.service("ALVideoDevice")
    fps = 20
    resolution = 2      # 0 == 160, 120 | 1 == 320, 240 | 2 == 640, 480
    colorSpace = 11     # 11 == bgr?
    name_id = video_service.subscribe("rgb_t", resolution, colorSpace, fps)

    motion = session.service("ALMotion")
    useSensorValues = False
    img_location = pd.DataFrame({"img_num": [0], "x_loc": [0], "y_loc": [0], "theta_loc": [0]})
    img_idx = 0

    tts = session.service("ALTextToSpeech")
    tts.setVolume(0.7)
    tts.setParameter("defaultVoiceSpeed", 65)

    #todo need to change query object before run code!!
    query_object = "doll"
    object_location = {'bleach': 'kitchen', 'boat': 'bedroom', 'controller': 'living room', 'tissue': 'living room',
                       'crocs': 'bedroom', 'doll': 'bedroom', 'doma': 'kitchen', 'hat': 'living room',
                       'picker': 'kitchen', 'pororo': 'living room', 'tayo': 'bedroom', 'can': 'kitchen'}

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    TCP_IP = '192.168.50.201'
    TCP_PORT = 5001
    sock.connect((TCP_IP, TCP_PORT))
    key_location = {}

    while True:
        direction = raw_input("Which direction do you want to move?")
        # i: forward, k: backward, j: left, l: right, q: quit
        if direction == "i":
            id = motion.moveTo(0.3, 0, 0)

            name_id = video_service.subscribe("rgb_t", resolution, colorSpace, fps)
            pepper_img = video_service.getImageRemote(name_id)

            img_idx += 1

            video_service.unsubscribe(name_id)
            width = pepper_img[0]
            height = pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)

            location = motion.getRobotPosition(useSensorValues)

            img_location.loc[img_location.shape[0]] = [img_idx, location[0], location[1], location[2]]

            motion.wait(id, 0)

            send_image(im, sock)
        if direction == "k":
            #id = motion.moveTo(-0.5, 0, 3.14)
            id = motion.moveTo(-0.3, 0, 0)

            name_id = video_service.subscribe("rgb_t", resolution, colorSpace, fps)
            pepper_img = video_service.getImageRemote(name_id)

            img_idx += 1

            video_service.unsubscribe(name_id)
            width = pepper_img[0]
            height = pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)

            location = motion.getRobotPosition(useSensorValues)

            img_location.loc[img_location.shape[0]] = [img_idx, location[0], location[1], location[2]]

            motion.wait(id, 0)

            send_image(im, sock)
        if direction == "j":
            #id = motion.moveTo(0, 0.5, 1.57)
            id = motion.moveTo(0, 0.3, 0)

            name_id = video_service.subscribe("rgb_t", resolution, colorSpace, fps)
            pepper_img = video_service.getImageRemote(name_id)

            img_idx += 1

            video_service.unsubscribe(name_id)
            width = pepper_img[0]
            height = pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)

            location = motion.getRobotPosition(useSensorValues)

            img_location.loc[img_location.shape[0]] = [img_idx, location[0], location[1], location[2]]

            motion.wait(id, 0)

            send_image(im, sock)
        if direction == "l":
            #id = motion.moveTo(0, -0.5, -1.57)
            id = motion.moveTo(0, -0.3, 0)

            name_id = video_service.subscribe("rgb_t", resolution, colorSpace, fps)
            pepper_img = video_service.getImageRemote(name_id)

            img_idx += 1

            video_service.unsubscribe(name_id)
            width = pepper_img[0]
            height = pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)

            location = motion.getRobotPosition(useSensorValues)

            img_location.loc[img_location.shape[0]] = [img_idx, location[0], location[1], location[2]]

            motion.wait(id, 0)

            send_image(im, sock)

        if direction == "o":
            #id = motion.moveTo(0, -0.5, -1.57)
            id = motion.moveTo(0, 0, -0.79)

        if direction == "u":
            #id = motion.moveTo(0, 0.5, 1.57)
            id = motion.moveTo(0, 0, 0.79)

        if direction == ",":
            #id = motion.moveTo(0, 0.5, 1.57)
            id = motion.moveTo(0, 0, 3.14)

        if direction == "1": #start position
            location = motion.getRobotPosition(useSensorValues)
            print("start position: ", location)
            #key_location["start"] = location

        if direction == "2": #kitchen postion
            location = motion.getRobotPosition(useSensorValues)
            key_location["kitchen"] = location

        if direction == "3": #living_room postion
            location = motion.getRobotPosition(useSensorValues)
            key_location["living room"] = location

        if direction == "4": #bedroom postion
            location = motion.getRobotPosition(useSensorValues)
            key_location["bedroom"] = location

        if direction == "5": #end position
            location = motion.getRobotPosition(useSensorValues)
            print("end position: ", location)
            #key_location["end"] = location

        if direction == "s": # save log data
            now = datetime.now()
            date_time = now.strftime("%m%d")
            img_dir = "/home/aupair/Desktop/One-Shot-Object-Detection/test/output/" + query_object + str(date_time)

            img_location_data = pd.DataFrame(img_location)
            img_location_data.to_csv(img_dir+ "/" + "location_info" + ".csv")
            im = 0
            send_image(im, sock)

        if direction == "r": # result
            score = pd.read_csv(img_dir+ "/" + "score_info" + ".csv")
            sorted_score = score.sort_values(by='cls_score', ascending = False)
            head_score = sorted_score.head(1)
            best_idx = int(head_score['img_num'])

            location_info = pd.read_csv(img_dir + "/" + "location_info" + ".csv")
            best_location = list(location_info.loc[best_idx, 'theta_loc':'y_loc'])
            print("Highest_score_img: ", best_idx, "location: ", best_location)

            distance = {}
            for key, val in key_location.items():
                dist = np.sqrt((val[0]-best_location[0])**2 + (val[1]-best_location[1])**2)
                distance[key] = dist
            print("Distances from: ", distance)

            sorted_distance = sorted(distance.items(), key = lambda x: x[1])
            print("Sorted_distances: ", sorted_distance)

            closest_location = sorted_distance[0][0]
            print("Closest_location: ", closest_location)

            if closest_location == object_location[query_object]:
                result_ = 'Success'
            else:
                result_ = 'Fail'

            result = {'Query Object': query_object, 'Highest Score Image': best_idx,
                      'Image Location Info': best_location, 'Pepper Guess Place': closest_location,
                      'Ground Truth Place': object_location[query_object], 'Result': result_}
            result_data = pd.Series(result)
            result_data.to_csv(img_dir + "/" + "result" + ".csv")

            print("I found the " + str(query_object) + ". " + "It is in " + str(closest_location))
            tts.say("I found the " + str(query_object) + ". " + "It is in " + str(closest_location))
            best_img = img.imread(img_dir + "/" + str(best_idx) + ".png")
            plt.imshow(best_img)
            plt.show()

        if direction == "q":
            break

    video_service.unsubscribe(name_id)

    sock.close()

### config ####
PEPPER_IP = '192.168.50.188'
parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default=PEPPER_IP,
                    help="Robot IP address. On robot or Local Naoqi: use '192.168.50.188'.")
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