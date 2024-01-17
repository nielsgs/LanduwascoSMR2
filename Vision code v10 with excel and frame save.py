import cv2
from ultralytics import YOLO
import numpy as np
import socket
import pandas as pd
import time
from datetime import datetime
import os
#pip install pandas openpyxl


HOST = "192.168.137.5"
PORT = 22400

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
frame_folder = f'detection/frames_visionV07-{current_time}/'
os.makedirs(frame_folder, exist_ok=True)

# Initialize variables
frame_count = 0
vision_round = 0
VL_passed = 0
SC_False = 0
SC_True = 0
start_time = time.time()
data = {'Vision Round': [], 'Time Elapsed (seconds)': [], 'VL_passed': [], "SC_False": [], 'SC_True': []}

def save_to_excel(dataframe):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_filename = f'detection/visionV07-{current_time}.xlsx'
    dataframe.to_excel(excel_filename, index=False)
    
    
def vision_test():
    global last_coordinates
    global vision_round     
    global VL_passed
    global SC_False
    global SC_True
    global frame_count
    
    vision_round += 1
    # Append data to dictionary
    elapsed_time = time.time() - start_time
    data['Vision Round'].append(vision_round)
    data['Time Elapsed (seconds)'].append(elapsed_time)
    data['VL_passed'].append(VL_passed)
    data['SC_False'].append(SC_False)
    data['SC_True'].append(SC_True)
    
    last_coordinates = False
    sent_coordinates = False

    virtual_line = 270
    conveyorwait = 1000
    confidence = 0.90

    # Load the YOLOv8 model
    model = YOLO('runs/detect/yolov8n_toweltrainingV43/weights/best.pt', 'v8')

    # Set pixel and world coordinates
    RW_X = 400
    RW_Y = 480
    pixel_X = 214
    pixel_Y = 255
    pix_RW_X = RW_X / pixel_X
    pix_RW_Y = RW_Y / pixel_Y
    x_C, y_C = 248, 128
    
    #set colors
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)
    pink = (92,11,227)
    orange = (5,94,255)
    purple = (255,25,162)
    black = (0,0,0)
    white = (255,255,255)
    grey = (200,200,200)

    thickness1 = 1
    thickness2 = 2
    size1 = 0.7
    size2 = 0.7
    d1 = 4
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    font2 = cv2.FONT_HERSHEY_SIMPLEX

    # Open the video file
    cap = cv2.VideoCapture(2)

    while cap.isOpened() and not last_coordinates:
        ret, frame_original = cap.read()

        if ret:
            results = model.detect(frame_original, show=True, conf=confidence)

            if results.xyxy[0].shape[0] > 0:
                # Object detected
                bbox = results.xyxy[0][0].cpu().numpy()
                x, y, w, h = bbox[:4]
                midpoint_x = int(x + w / 2)
                midpoint_y = int(y + h / 2)
                
                x_P, y_P = midpoint_x, midpoint_y
                cv2.circle(frame_original,(x_C,y_C),10,(blue),6)
                cv2.circle(frame_original,(x_P,y_P),6,(red),6)

                # Calculate relative movement in pixels
                xrev_pixel = midpoint_x - x_C
                yrev_pixel = midpoint_y - y_C

                # Calculate relative movement in real world
                xrev_wereld = int(xrev_pixel * pix_RW_X)
                yrev_wereld = int(yrev_pixel * pix_RW_Y)
                xrev_wereld = str(xrev_wereld)
                yrev_wereld = str(yrev_wereld)

                print("Move:", (yrev_wereld, xrev_wereld))

                # Send coordinates only if they haven't been sent before
                if midpoint_x >= virtual_line:
                    
                    belt_off = str(-5)
                    rev_move = f'{belt_off}'  # Combine DY, DX, and belt_off values
                    client_socket.sendall(rev_move.encode("utf-8"))

                    sent_coordinates=False
                    
                    VL_passed=1
                    # Append data to dictionary
                    elapsed_time = time.time() - start_time
                    data['Vision Round'].append(vision_round)
                    data['Time Elapsed (seconds)'].append(elapsed_time)
                    data['VL_passed'].append(VL_passed)
                    data['SC_False'].append(SC_False)
                    data['SC_True'].append(SC_True)
                    
                    
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")
                    print("PASSED VIRTUALLINE")

                    cv2.waitKey(conveyorwait)

                    while sent_coordinates==False:
                        ret, frame_original = cap.read()
                        
                        SC_False +=1
                        # Append data to dictionary
                        elapsed_time = time.time() - start_time
                        data['Vision Round'].append(vision_round)
                        data['Time Elapsed (seconds)'].append(elapsed_time)
                        data['VL_passed'].append(VL_passed)
                        data['SC_False'].append(SC_False)
                        data['SC_True'].append(SC_True)

                        print("while loop sent_coordinates == False loop")
                        print("while loop sent_coordinates == False loop")
                        print("while loop sent_coordinates == False loop")
                        print("while loop sent_coordinates == False loop")
                        print("while loop sent_coordinates == False loop")
                        print("while loop sent_coordinates == False loop")

                        if ret:
                            results = model.detect(frame_original, show=True, conf=confidence)

                            if results.xyxy[0].shape[0] > 0:
                                # Object detected
                                sent_coordinates = True
                                
                                SC_True = 1
                                SC_False=0
                                # Append data to dictionary
                                elapsed_time = time.time() - start_time
                                data['Vision Round'].append(vision_round)
                                data['Time Elapsed (seconds)'].append(elapsed_time)
                                data['VL_passed'].append(VL_passed)
                                data['SC_False'].append(SC_False)
                                data['SC_True'].append(SC_True)
                                
                                
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")
                                print("object detected: sent_coordinates set to TRUE")

                                bbox = results.xyxy[0][0].cpu().numpy()
                                x, y, w, h = bbox[:4]
                                midpoint_x = int(x + w / 2)
                                midpoint_y = int(y + h / 2)

                                # Calculate relative movement in pixels
                                xrev_pixel = midpoint_x - x_C
                                yrev_pixel = midpoint_y - y_C
                                
                                x_P, y_P = midpoint_x, midpoint_y
                                cv2.circle(frame_original,(x_C,y_C),10,(blue),3)
                                cv2.circle(frame_original,(x_P,y_P),6,(red),2)

                                # Calculate relative movement in real world
                                xrev_wereld = int(xrev_pixel * pix_RW_X)
                                yrev_wereld = int(yrev_pixel * pix_RW_Y)
                                xrev_wereld = str(xrev_wereld)
                                yrev_wereld = str(yrev_wereld)

                                print("Move:", (yrev_wereld, xrev_wereld))
                                a = f'{yrev_wereld},{xrev_wereld}'  # Combine DY, DX, and belt_off values
                                client_socket.sendall(a.encode("utf-8"))
                                
                                # Save the frame with a specific name
                                frame_name = f'Towel_{vision_round}_frame{frame_count}.jpg'
                                frame_path = os.path.join(frame_folder, frame_name)
                                cv2.imwrite(frame_path, frame_original)
                    
                                frame_count += 1

                                last_coordinates = True
                                break

            # Display the frame
            cv2.imshow("YOLOv8 Detection", frame_original)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()
    cv2.destroyAllWindows()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    message_to_server = "Hello world, this is spyder"
    client_socket.sendall(message_to_server.encode("utf-8"))

    while True:
        print("Waiting for command...")

        data = client_socket.recv(1024)
        server_response = data.decode("utf-8")
        print("Server response:", server_response)

        if server_response == 'exit':
            print('exiting...')
            break
        elif server_response == 'vision_test()':
            vision_test()
            
# Create and save the DataFrame after the loop ends
df = pd.DataFrame(data)
save_to_excel(df)
