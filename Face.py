import face_recognition
import cv2
import time


def read_video(video_name):
    cap = cv2.VideoCapture(video_name)

    count = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        save_path = './test/{:>03d}.jpg'.format(count)
        cv2.imwrite(save_path, frame)
        count += 1
    # cv2.waitKey(0)
    return cap


def parse_video():
    for index in range(1, 331):
        file_name = './test/' + str(index).rjust(3, '0') + '.jpg'
        _face_landmarks_dict = parse_pic(file_name)
        draw_pic(_face_landmarks_dict, file_name)


def parse_pic(file_name):
    image = face_recognition.load_image_file(file_name)
    face_landmarks_list = face_recognition.face_landmarks(image)
    # for one person
    face_landmarks_dict = face_landmarks_list[0]
    return face_landmarks_dict


def draw_pic(face_landmarks_dict, file_name):
    img = cv2.imread(file_name)
    white = (255, 255, 255)
    values = list(face_landmarks_dict.values())
    # print(values)
    for each in values:
        for i in range(len(each) - 1):
            cv2.line(img, each[i], each[i + 1], white, 1)
    # cv2.namedWindow(file_name)
    # cv2.imshow(file_name, img)
    # print(file_name)
    cv2.imwrite(file_name.split('.jpg')[0] + '_after.jpg', img)
    # cv2.waitKey(0)





def make_video(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),\
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter('./test/video.avi', \
                                   cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for i in range(1, 331):
        file_name = './test/' + str(i).rjust(3, '0') + '_after.jpg'
        img = cv2.imread(file_name)
        video_writer.write(img)
    cap.release()

if __name__ == '__main__':
    '''
    image = face_recognition.load_image_file(file_name)
    t1=time.time()
    face_landmarks_list = face_recognition.face_landmarks(image)
    t2=time.time()
    print('cost is {}s.'.format(t2-t1))
    print(face_landmarks_list)
    face_landmarks_dict=face_landmarks_list[0]
    '''
    video_name = 'BillClinton.avi'
    cap = read_video(video_name)
    parse_video()
    make_video(cap)

    # draw_pic(face_landmarks_dict)
    '''
    [{'chin': [(67, 51), (67, 58), (67, 64), (67, 70), (69, 76), (72, 80), (77, 84), (83, 87), (89, 89), (95, 89),
               (100, 87), (104, 83), (107, 79), (109, 74), (110, 69), (111, 64), (112, 59)],
      'left_eyebrow': [(77, 46), (80, 44), (84, 43), (89, 44), (92, 46)],
      'right_eyebrow': [(98, 48), (102, 48), (106, 48), (109, 50), (110, 53)],
      'nose_bridge': [(95, 51), (95, 55), (95, 59), (95, 63)],
      'nose_tip': [(88, 65), (90, 66), (93, 67), (95, 67), (97, 67)],
      'left_eye': [(81, 50), (84, 49), (86, 50), (88, 52), (86, 52), (83, 51)],
      'right_eye': [(99, 54), (102, 54), (104, 54), (106, 56), (104, 56), (102, 55)],
      'top_lip': [(82, 72), (86, 71), (90, 71), (92, 72), (94, 72), (97, 73), (100, 75), (98, 75), (94, 74), (92, 73),
                  (89, 73), (83, 72)],
      'bottom_lip': [(100, 75), (96, 76), (94, 76), (91, 75), (88, 75), (85, 74), (82, 72), (83, 72), (89, 72),
                     (91, 73), (94, 73), (98, 75)]}]
    '''
    # [{'nose_tip': [(476, 201)], 'left_eye': [(455, 175), (469, 178)], 'right_eye': [(505, 176), (489, 178)]}]

    # [[(476, 201)], [(455, 175), (469, 178)], [(505, 176), (489, 178)]]
    '''
    face_loca = [i for i in face_locations[0].values()]
    print(face_loca)
    # draw eyes
    green = (0, 255, 0)
    cv2.line(img, face_loca[1][0], face_loca[1][1], green, 2)
    cv2.line(img, face_loca[2][0], face_loca[2][1], green, 2)
    red=(0,0,255)
    cv2.line(img, face_loca[0][0], face_loca[0][0], red, 2)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    '''
