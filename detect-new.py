import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import mediapipe as mp

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                   exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    isDrawing = False
    far_points = []
    tempFarPoints = []
    frameCounter = 0
    # Run inference
    t0 = time.time()
    pressed_key = cv2.waitKey(1)
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # experiment
        # img = cv2.flip(np.float32(img) , 1)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' %
                                                            dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            croppedImage = im0s[i].copy()
            image = im0.copy()  # print string
            # experiment
            # image = cv2.flip(np.float32(image) , 1)
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                xyxy = det[:, :4]
                xyxy = xyxy.flatten()
                xyxy = xyxy.tolist()

                xyxyInt = []
                for items in xyxy:
                    xyxyInt.append(int(items))
                kernel_size = 5
                kernel1 = np.ones((kernel_size, kernel_size),
                                  np.float32)/kernel_size/kernel_size
                kernel2 = np.ones((10, 10), np.uint8)/100
                print(xyxyInt)
                croppedImage = croppedImage[xyxyInt[1]
                    :xyxyInt[3], xyxyInt[0]:xyxyInt[2]]
                
                #after detection
                min_YCrCb = np.array([0, 40, 50], np.uint8)
                max_YCrCb = np.array([50, 250, 255], np.uint8)
                imageYCrCb = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
                res = cv2.erode(mask, kernel1, iterations=1)
                res = cv2.dilate(res, kernel1, iterations=1)
                res = cv2.bitwise_and(imageYCrCb, imageYCrCb, mask=mask)

                rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
                cv2.imshow('rgb_2', rgb)
                # cv2.imshow('rgb',rgb)
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('gray',gray)
                # gray = cv2.filter2D(gray,-1,kernel2)    # hacky
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
                cv2.imshow('gray', gray)
# Find region with skin tone in YCrCb image
                contours, hierarchy = cv2.findContours(
                    gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = max(contours, key=lambda x: cv2.contourArea(x))
                hull = cv2.convexHull(contours, returnPoints=False)
                # defects = cv2.convexityDefects(contours, hull)
                # Print results
                c = contours
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                extLeft1 = (extLeft[0]+xyxyInt[0], extLeft[1]+xyxyInt[1])
                extRight1 = (extRight[0]+xyxyInt[0], extRight[1]+xyxyInt[1])
                extTop1 = (extTop[0]+xyxyInt[0], extTop[1]+xyxyInt[1])
                extBot1 = (extBot[0]+xyxyInt[0], extBot[1]+xyxyInt[1])
                cv2.circle(image, extLeft1, 8, (0, 0, 255), -1)
                cv2.circle(image, extRight1, 8, (0, 255, 0), -1)
                cv2.circle(image, extTop1, 8, (255, 0, 0), -1)
                cv2.circle(image, extBot1, 8, (255, 255, 0), -1)

                if pressed_key & 0xFF == ord('x'):
                    far_points.clear()
                classDetected = det[:, -1].tolist()
                classDetected = int(classDetected[0])
                if classDetected == 0:
                    isDrawing = True
                    tempFarPoints.append(extTop1)

                elif classDetected == 1:
                    #after cropping

                    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:

                        # BGR 2 RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Flip on horizontal
                        image = cv2.flip(image, 1)

                        # Set flag
                        image.flags.writeable = False

                        # Detections
                        results = hands.process(image)

                        # Set flag to true
                        image.flags.writeable = True

                        # RGB 2 BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Detections
                        print(results)
                        image_height, image_width, _ = image.shape

                        # Rendering results
                        if results.multi_hand_landmarks:
                            for num, hand in enumerate(results.multi_hand_landmarks):
                                for hand_landmarks in results.multi_hand_landmarks:
                                    #                     print('hand_landmarks:', hand_landmarks)
                                    far_points.append((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)))

                                    for j in range(len(far_points) - 1):
                                        cv2.line(image, far_points[j],
                                                far_points[j+1], (255, 5, 255), 10)
                                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                        mp_drawing.DrawingSpec(
                                                            color=(121, 22, 76), thickness=2, circle_radius=4),
                                                        mp_drawing.DrawingSpec(
                                                            color=(250, 44, 250), thickness=2, circle_radius=2),
                                                        )


                    # old rendering code
                    # isDrawing = False
                    # if len(tempFarPoints) != 0:
                    #     far_points.append(tempFarPoints)
                    #     tempFarPoints = []
                    #     frameCounter = frameCounter + 1
                    #     if(frameCounter <= 15):
                    #         tempArray = []
                    #         index = len(far_points) - 1
                    #         print(range(len(far_points[index])))
                    #         for i in range(len(far_points[index])):
                    #             print(
                    #                 ".asnkfk,abkjaskdbkjadbsbvkjbasj.bvbbkjvbjebfjk.bNSDb,")
                    #             if i == 0 or i == len(far_points[index]) - 1:
                    #                 tempArray.append(far_points[index][i])
                    #             else:
                    #                 avgx = int(
                    #                     (far_points[index][i-1][0] + far_points[index][i+1][0])/2)
                    #                 avgy = int(
                    #                     (far_points[index][i-1][1] + far_points[index][i+1][1])/2)
                    #                 tempArray.append((avgx, avgy))
                    #         far_points[index] = tempArray
                    #     else:
                    #         frameCounter = 0

                # cv2.line(canvas, far_points[i], far_points[i+1], (0,0,0), 20)
                # if isDrawing:
                #     # tempFarPoints.append(extTop1)
                #     # far_points.append(extTop1)

                # if len(far_points) == 0 or isDrawing:
                #     for i in range(len(tempFarPoints)-1):
                #         cv2.line(image, tempFarPoints[i],
                #                  tempFarPoints[i+1], (255, 5, 255), 10)

                # if len(far_points) != 0 or isDrawing:
                #     for i in range(len(far_points)-1):
                #         for j in range(len(far_points[i]) - 1):
                #             cv2.line(image, far_points[i][j],
                #                      far_points[i][j+1], (255, 5, 255), 10)

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (
                            cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), image)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
