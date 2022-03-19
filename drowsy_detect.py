# YOLOv5 Ã°Å¸Å¡â‚¬ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

color_move_on = (255, 200, 90)
color_red = (25, 20, 240)
color = color_move_on
text_x_align = 10
inference_time_y = 30
fps_y = 90
analysis_time_y = 60
font_scale = 0.7
thickness = 2
rect_thickness = 2
tl = rect_thickness
total_arr = []
counter2 = 0
counter = 0


@torch.no_grad()
def run(weights=ROOT / 'weights/yolov5l.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    global counter2
    global counter
    global total_arr
    length = 0
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            # check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            normal_arr = []
            drowsy_arr = []
            drowsy2_arr = []
            yawning_arr = []

            # normal_num = 0
            # drowsy_num = 0
            # drowsy2_num = 0

            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                color = (255, 200, 90)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.box_label(xyxy, label, color=colors(c, False))

                        if names[c] == 'normal':
                            color = (255, 200, 90)
                        elif names[c] == 'drowsy':
                            color = (0, 0, 255)
                        elif names[c] == 'drowsy#2':
                            color = (0, 0, 255)
                        elif names[c] == 'yawning':
                            color = (51, 255, 255)

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        if names[c]:
                            if names[c] == 'normal':
                                normal_arr.append([names[c]])
                                drowsy_text = 'NORMAL '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (255, 200, 90),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)
                            elif names[c] == 'drowsy':
                                drowsy_arr.append([names[c]])
                                s_img = cv2.imread("icon/warning.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'WARNING '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (0, 0, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            elif names[c] == 'drowsy#2':
                                drowsy2_arr.append([names[c]])
                                s_img = cv2.imread("icon/warning.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'WARNING '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (0, 0, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            elif names[c] == 'yawning':
                                yawning_arr.append([names[c]])
                                s_img = cv2.imread("icon/coffe2.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'DROWSY '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (51, 255, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            lab_text = '{} {:0.2f} '.format(names[c],conf)
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(lab_text, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            cv2.putText(im0, lab_text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf,
                                        lineType=cv2.LINE_AA)

                            normal_num = len(normal_arr)
                            drowsy_num = len(drowsy_arr)
                            drowsy2_num = len(drowsy2_arr)
                            yawning_num = len(yawning_arr)

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                k = cv2.waitKey(0) & 0xFF

                if k == 27:  # wsc
                    cv2.destroyAllWindows()

            # # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './weights/yolov5l.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
