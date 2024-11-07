#import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import onnxruntime as ort
import serial
import func as f
ser = serial.Serial('/dev/ttyS0', 115200)


if __name__ == "__main__":
    #发送列表
    send =['a','b','c','d']
    #开始
    #rem为一开始所得到的数字,send为运动指示
    rem,send[0] = f.Start()

    #发送我们的指令
    #在这个地方一般只有前进，向左向右
    ser.write(send[0].encode())
    time.sleep(0.3)

    if(rem == 1 |rem == 2):
        f.find_stop()
        re = 'S'
        #发送停车指令
        ser.write(re.encode())
        time.sleep(0.3)

    else:
        #将一开始所得到的数字带入二号路口的检测之中
        #这个时候接收的是运动的方向指示
        #G直行，L左转，R右转
        send[1] = f.jud1(rem)

        ###判断是中端还是远端###

        #这个时候是中段
        if send[1] == 'L' | send[1] == 'R':
            #发送我们的识别指令
            ser.write(send[1].encode())
            time.sleep(0.3)

            #开始寻找刹车点
            f.find_stop()
            re = 'S'
            # 发送停车指令
            ser.write(re.encode())
            time.sleep(0.3)


            ############新增###########
            ###返程刹车
            time.sleep(8)
            f.find_stop()
            re = 'S'
            # 发送停车指令
            ser.write(re.encode())
            time.sleep(0.3)



        ###################新增################
        #这个时候是远端
        else:
            #发送前进的指令
            ser.write(send[1].encode())
            time.sleep(4)

            ###进行四个数字的大路口检测
            send[2] = f.jud2(rem)

            ser.write(send[2].encode())
            time.sleep(0.2)

            time.sleep(2)

            ###进行丁字路口检测
            send[3] = f.jud1(rem)

            ser.write(send[3].encode())
            time.sleep(0.2)

            # 发送停车指令
            f.find_stop()
            re = 'S'

            ser.write(re.encode())
            time.sleep(0.3)

            time.sleep(10)
            # 发送最后的停车指令
            f.find_stop()
            re = 'S'

            ser.write(re.encode())
            time.sleep(0.3)




