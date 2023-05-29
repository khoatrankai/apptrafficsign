
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, is_ascii, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from PIL import ImageFont, Image, ImageDraw, ImageTk
from pathlib import Path
import matplotlib.font_manager as fm
import torch
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from functools import partial
global window
window = tk.Tk()
global label21
label21 = tk.Label(window)
global label22
label22 = tk.Label(window)


global hcn3
hcn3 = tk.Canvas(window, width=600, height=500)

global hcn4
hcn4 = tk.Canvas(window, width=600, height=500)

global hcn5
hcn5 = tk.Canvas(window, width=280, height=500)

global checkgg
checkgg = True
global checkloai
checkloai = True

global label_train
label_train = tk.Label(window)

# global label22
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
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
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith(
        '.streams') or (is_url and not is_file)

    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    # Load model
    device = select_device(device)
    # Lấy dữ liệu trong best.pt ra để render giao diện
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("names: ", names)
    global loaibienbaos
    loaibienbaos = ["-Biển báo cấm", "-Biển báo nguy hiểm",
                    "-Biển báo chỉ dẫn", "-Biển báo tốc độ"]
    global thuocbienbao
    thuocbienbao = [
        0, 0, 2, 2, 1, 1, 0, 1, 1, 1, 0, 2, 0, 0, 1, 1, 0, 0, 3, 0, 0, 0, 2, 2, 0, 2, 1
    ]
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)

        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)

    else:
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        if checkgg and checkloai:
            break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions (dự đoán dữ liệu đúng có trong ảnh)
        for i, det in enumerate(pred):  # per image
            # if checkgg == False:
            #     break
            # print("128: ")
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            cdimage_pil1 = Image.fromarray(
                cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            cdimage_pil1 = cdimage_pil1.resize((600, 500), Image.LANCZOS)
            cdimage_tk1 = ImageTk.PhotoImage(cdimage_pil1)
            label21.configure(image=cdimage_tk1)
            label21.image = cdimage_tk1
            if len(det):
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()
                im0 = design_det(im0, det[:, :6], names)
            cdimage_pil = Image.fromarray(
                cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            cdimage_pil = cdimage_pil.resize((600, 500), Image.LANCZOS)
            cdimage_tk = ImageTk.PhotoImage(cdimage_pil)
            label22.configure(image=cdimage_tk)
            label22.image = cdimage_tk
            window.update()


def set_checkgg():

    global checkgg
    if checkgg:
        checkgg = False
    else:
        checkgg = True


def set_checkloai1():
    global checkloai
    checkloai = True


def set_checkloai2():
    global checkloai
    checkloai = False


def design_det(img, det, names):

    allname = ''
    alltraffic = ''
    # tạo khung và thêm label tên vào ảnh
    mangloaibb = [0]*(len(names))
    for i in range(det.shape[0]):
        # lấy tọa độ điểm nhận dạng

        toadox = int(det[i, 0].item())
        toadoy = int(det[i, 1].item())
        toadow = int(det[i, 2].item())
        toadoh = int(det[i, 3].item())
        phantram = float(det[i, 4].item())
        diem = int(det[i, 5].item())

        if phantram > 0.5:
            # vẽ khung các điểm nhận dạng được
            cv2.rectangle(img, (toadox, toadoy),
                          (toadow, toadoh), (0, 255, 0), 2)
            # lấy tên từng khung điểm để xuất label
            text = names[diem]
            mangloaibb[diem] = mangloaibb[diem] + 1
            loaibb = loaibienbaos[thuocbienbao[diem]]
            if allname == '':
                allname = text
            else:
                allname = allname + "," + text
            font = ImageFont.truetype("arial.ttf", 24)
            bbox = font.getbbox(text+loaibb)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # tạo chữ thành ảnh
            image = Image.new(mode='RGB', size=(w, h), color=(0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), text+loaibb, font=font, fill=(255, 255, 255))
            numpy_image = np.array(image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            rows, cols, channels = opencv_image.shape
            # cắt tọa độ điểm trong ảnh để thêm ảnh chữ vào
            labeltext = img[toadoy:toadoy+rows, toadox:toadox+cols]
            # chống bị resize giữa 2 điểm ảnh bị lệch
            gray_image = cv2.cvtColor(labeltext, cv2.COLOR_BGR2GRAY)
            opencv_image = cv2.resize(
                opencv_image, (gray_image.shape[1], gray_image.shape[0]))
            # hợp nhất 2 điểm ảnh giữa ảnh gốc và các điểm có ảnh chữ
            labeltext = cv2.addWeighted(labeltext, 1, opencv_image, 1, 0)
            img[toadoy:toadoy+rows, toadox:toadox+cols] = labeltext

    for i in range(len(mangloaibb)):
        if mangloaibb[i] != 0:
            alltraffic = alltraffic + \
                str(mangloaibb[i])+" : " + names[i] + '\n'
    xulychulabel(alltraffic)
    return img


def traffic_sign_cam():
    # print(window)
    # Tạo một hình ảnh từ tệp hình ảnh
    set_checkloai1()
    btn3.place(x=600, y=650)
    btn1.place_forget()
    btn2.place_forget()
    hcn3.place_forget()
    hcn4.place_forget()
    hcn5.place_forget()
    label21.place(x=10, y=105)
    label22.place(x=620, y=105)
    label_train.place(x=1240, y=105)
    opt = argparse.Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, data=('data/coco128.yaml'), device='', dnn=False, exist_ok=False, half=False, hide_conf=False, hide_labels=False, imgsz=[
                             640, 640], iou_thres=0.45, line_thickness=3, max_det=1000, name='exp', nosave=False, project=('runs/detect'), save_conf=False, save_crop=False, save_txt=False, source='0', update=False, vid_stride=1, view_img=False, visualize=False, weights=['best9_5_2.pt'])

    set_checkgg()
    main(opt)


def set_label_train():
    imgblack = Image.new(mode='RGB', size=(280, 500), color=(0, 0, 0))
    photo_image = ImageTk.PhotoImage(imgblack)
    label_train.configure(image=photo_image)
    label_train.image = photo_image


def xulychulabel(text):
    font = ImageFont.truetype("arial.ttf", 20)
    w = 280
    h = 500
    # tạo chữ thành ảnh
    image = Image.new(mode='RGB', size=(w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    photo_image = ImageTk.PhotoImage(image)
    label_train.configure(image=photo_image)
    label_train.image = photo_image


def upload_train():
    filepath = filedialog.askopenfilename()

    # Kiểm tra xem người dùng đã chọn một tập tin hay chưa
    if filepath:
        hcn5.place_forget()
        hcn3.place_forget()
        hcn4.place_forget()
        label_train.place(x=1240, y=105)
        label21.place(x=10, y=105)
        label22.place(x=620, y=105)
        # Mở ảnh với PIL
        opt = argparse.Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, data=('data/coco128.yaml'), device='', dnn=False, exist_ok=False, half=False, hide_conf=False, hide_labels=False, imgsz=[
            640, 640], iou_thres=0.45, line_thickness=3, max_det=1000, name='exp', nosave=False, project=('runs/detect'), save_conf=False, save_crop=False, save_txt=False, source=filepath, update=False, vid_stride=1, view_img=False, visualize=False, weights=['best9_5_2.pt'])
        main(opt)


def choose_close():
    set_label_train()
    if checkloai:
        set_checkgg()
    hcn3.place(x=10, y=105)
    hcn4.place(x=620, y=105)
    hcn5.place(x=1240, y=105)
    btn1.place(x=180, y=650)
    btn2.place(x=1000, y=650)
    btn3.place_forget()
    btn_upload.place_forget()
    label21.place_forget()
    label22.place_forget()
    label_train.place_forget()


def traffic_sign_image():
    global btn_upload
    btn_upload = tk.Button(window, text="UPLOAD",
                           command=upload_train, width=20, height=6, font=("Arial", 15, "bold"), activebackground="#FF3131", activeforeground="#FFFFFF")
    btn_upload.place(x=180, y=650)
    set_checkloai2()
    label_train.place(x=1240, y=105)
    btn3.place(x=1000, y=650)
    btn1.place_forget()
    btn2.place_forget()


def close_app():
    if messagebox.askokcancel("Đóng", "Bạn có chắc chắn muốn thoát?"):
        window.destroy()


def giaodien():
    window.geometry('1600x900')
    window.overrideredirect(True)
    window.state('zoomed')
    window.configure(background="#00BF63")
    # giao diện đầu
    hcn1 = tk.Canvas(window, width=1600, height=80)
    react = hcn1.create_rectangle(
        0, 0, 1600, 108, fill='black')
    hcn1.place(x=0, y=0)
    hcn2 = tk.Canvas(window, width=1600, height=300)
    react2 = hcn2.create_rectangle(
        0, 0, 1600, 500, fill='black')
    hcn2.place(x=0, y=620)

    react3 = hcn3.create_rectangle(
        0, 0, 1600, 1600, fill='black')
    hcn3.place(x=10, y=105)

    react4 = hcn4.create_rectangle(
        0, 0, 1600, 1600, fill='black')
    hcn4.place(x=620, y=105)

    react5 = hcn5.create_rectangle(
        0, 0, 1600, 1600, fill='black')
    hcn5.place(x=1240, y=105)

    global btn1
    btn1 = tk.Button(window, text="CAMERA",
                     command=traffic_sign_cam, width=20, height=6, activebackground="#FF3131", activeforeground="#FFFFFF", font=("Arial", 15, "bold"))
    btn1.place(x=180, y=650)

    global btn2
    btn2 = tk.Button(window, text="IMAGE",
                     command=traffic_sign_image, width=20, height=6, activebackground="#FF3131", activeforeground="#FFFFFF", font=("Arial", 15, "bold"))
    btn2.place(x=1000, y=650)

    global btn3
    btn3 = tk.Button(window, text="CLOSE",
                     command=choose_close, width=20, height=6, activebackground="#FF3131", activeforeground="#FFFFFF", font=("Arial", 15, "bold"))

    linkclose = './close2.png'
    imgclose = Image.open(linkclose)
    imgclose = ImageTk.PhotoImage(imgclose)
    label_close_app = tk.Label(
        window, image=imgclose)
    label_close_app.bind("<Button-1>", lambda event: close_app())
    label_close_app.place(x=1450, y=20)

    linklogo = './HCMUTE.png'
    imglogo = Image.open(linklogo)
    imglogo = ImageTk.PhotoImage(imglogo)
    label_logo = tk.Label(
        window, image=imglogo)
    label_logo.place(x=0, y=0)

    linktitle = './title.png'
    imgtitle = Image.open(linktitle)
    imgtitle = ImageTk.PhotoImage(imgtitle)
    label_title = tk.Label(
        window, image=imgtitle)
    label_title.place(x=80, y=0)

    linkteam = './team3.png'
    imgteam = Image.open(linkteam)
    imgteam = ImageTk.PhotoImage(imgteam)
    label_team = tk.Label(
        window, image=imgteam)
    label_team.place(x=880, y=0)

    set_label_train()
    window.mainloop()


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    giaodien()
