import cv2
import numpy as np
import onnxruntime as ort


def Start():
    model_path = "best.onnx"
    # 创建一个session的对话
    so = ort.SessionOptions()
    # 加载出对应的网络模型
    net = ort.InferenceSession(model_path, so)

    # 建立自己的标签字典
    dic_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}

    # 模型参数（以yolov5-lite为例）
    # 模型训练时设置的同一参数
    model_h = 320
    model_w = 320
    # 输出层数
    nl = 3
    # 锚框数
    na = 3
    # 缩小因子
    stride = [8., 16., 32.]
    # 锚框的具体数值
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)  # 锚框的转置

    cap = cv2.VideoCapture(0)
    rem = 101
    while True:
        jud, img0 = cap.read()

        det_boxes, scores, ids = Process_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4,
                                             thred_cond=0.5)

        for box, score, id in zip(det_boxes, scores, ids):
            rem = id + 1

        if rem < 100:
            break
    cap.release()
    cv2.destroyAllWindows()
    if rem == 1:
        re = 'L'
    elif rem == 2:
        re = 'R'
    else:
        re = 'G'

    return rem, re


####################双数字检测函数###############
# 返回值:下位机所需要执行的指令
def jud1(number):
    # 一定是一次双数字的检测，将检测到的数字与输入进来的进行对比比较并判断位置，之后再进行判断
    model_path = "best.onnx"
    # 创建一个session的对话
    so = ort.SessionOptions()
    # 加载出对应的网络模型
    net = ort.InferenceSession(model_path, so)

    # 建立自己的标签字典
    dic_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}

    # 模型参数（以yolov5-lite为例）
    # 模型训练时设置的同一参数
    model_h = 320
    model_w = 320
    # 输出层数
    nl = 3
    # 锚框数
    na = 3
    # 缩小因子
    stride = [8., 16., 32.]
    # 锚框的具体数值
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)  # 锚框的转置

    cap = cv2.VideoCapture(0)
    rem = 101
    # 计算到底检测到了几个数字如果少于两个则重新开始
    count = 0
    ret = 'G'
    while True:
        jud, img0 = cap.read()

        det_boxes, scores, ids = Process_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4,
                                             thred_cond=0.5)

        for box, score, id in zip(det_boxes, scores, ids):
            # 解算出检测框的坐标
            x = box.astype(np.int16)
            count = count + 1

            # x[0]为每次检测的左上角坐标
            # 对比两个点的左上角坐标来判断左右的关系
            rem = id + 1
            if rem == number:
                if (x[0] < 320):
                    ret = 'L'
                else:
                    ret = 'R'

        if rem < 100 and count == 2:
            break

    cap.release()
    cv2.destroyAllWindows()
    return ret



def jud2(number):
    # 一定是一次四个数字的检测，将检测到的数字与输入进来的进行对比比较并判断位置，之后再进行判断
    model_path = "best.onnx"
    # 创建一个session的对话
    so = ort.SessionOptions()
    # 加载出对应的网络模型
    net = ort.InferenceSession(model_path, so)

    # 建立自己的标签字典
    dic_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}

    # 模型参数（以yolov5-lite为例）
    # 模型训练时设置的同一参数
    model_h = 320
    model_w = 320
    # 输出层数
    nl = 3
    # 锚框数
    na = 3
    # 缩小因子
    stride = [8., 16., 32.]
    # 锚框的具体数值
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)  # 锚框的转置

    cap = cv2.VideoCapture(0)
    rem = 101
    # 计算到底检测到了几个数字如果少于两个则重新开始
    count = 0
    ret = 'G'
    while True:
        jud, img0 = cap.read()
        img0 = img0[0:240, 0:320]
        det_boxes, scores, ids = Process_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4,
                                             thred_cond=0.5)

        for box, score, id in zip(det_boxes, scores, ids):
            # 解算出检测框的坐标
            x = box.astype(np.int16)
            count = count + 1

            # x[0]为每次检测的左上角坐标
            # 对比两个点的左上角坐标来判断左右的关系
            rem = id + 1
            #只要有相同的数字，直接拐
            if rem == number:
                    ret = 'L'

        if rem < 100 and count == 2:
            break

    if ret == 'G':
        ret = 'R'
    else:
        ret = 'L'


    cap.release()
    cv2.destroyAllWindows()
    return ret




###############停车函数###########
# 返回值：没有返回值，但是检测到之后才会结束函数
def find_stop():
    cap = cv2.VideoCapture(0)
    while True:
        jud, img = cap.read()
        img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        low_black = np.array([0, 0, 0])
        high_black = np.array([180, 255, 46])

        mask_black = cv2.inRange(img_cvt, low_black, high_black)

        contours, hir = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for i, contour in enumerate(contours):
            # 计算各块的面积
            area = cv2.contourArea(contour)
            # 通过面积的大小来进行筛选
            if area > 1800:
                cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
                count = count + 1

        if count >= 4:
            break

        cv2.imshow("img", mask_black)
        cv2.waitKey(1)


###############cal out#########################################
##注：直接通过神经网络所得到的数据是归一化之后的
def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def outputs_corr(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs


##################find the real number#######################
def find_realnumber(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []


############IMG Process###################
# function:find the id and place of number by our net
# include:outputs_corr,find_realnumber,_make_grid
def Process_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, (model_w, model_h), interpolation=cv2.INTER_AREA)
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # 归一化
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = outputs_corr(outs, nl, na, model_w, model_h, anchor_grid, stride)

    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = find_realnumber(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids


def Video():
    cap = cv2.VideoCapture(0)
    while True:
        jud, frame = cap.read()

        if not jud:
            print("Camera is not working")
            break

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

