Vision detection and tracking model.py
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

import socket

HOST = "192.168.137.5"
PORT = 22400

""" TODO / Improvements

    - Clean up code and use functions
    - apply crop on only the belt (changes coordinates)
    - try out downsampling frames 
    - use GPU
    - Try out object tracking with speed estimation for noise reduction
"""

def vision_test():
    
    #easy acces variables for testing
    global last_coordinates
    last_coordinates = False
    sent_coordinates = False
    i = 1 #track_id ==1 then i+1
    j = 1 #coordinates send then j+1
    virtual_line = 270
    conveyorwait = 1000
    confidence = 0.90
    track_time = 1 #only higher when using track_history information. otherwise not necessary to store old data
    
    track_history = defaultdict(lambda: [])  # Store the track history, create a defaultdict with default value as an empty list
    
    # Load the YOLOv8 model
    #model =YOLO('runs/detect/yolov8n_towel_detection_tracking/weights/best.pt', 'v8')
    model =YOLO('runs/detect/yolov8n_toweltrainingV43/weights/best.pt', 'v8')
    #model =YOLO('runs/detect/yolov8n_toweltrainingV733/weights/best.pt', 'v8')
    #model = YOLO('runs/detect/yolov8n_toweltrainingV43/weights/best.pt', 'v8', device='cuda')
    
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

    #set pixel and world coordinates
    RW_X = 400
    RW_Y = 480
    pixel_X = 214
    pixel_Y = 255
    pix_RW_X = (RW_X/pixel_X)   #1 pixel naar real world coordinates 
    pix_RW_Y = (RW_Y/pixel_Y)
    x_C,y_C = 248,128  #nulpunt
    
    # Open the video file
    cap = cv2.VideoCapture(2)
    
    # Loop through the video frames
    while cap.isOpened() and last_coordinates == False:
        ret, frame_original = cap.read()
        #frame_original = frame_original[110:365,0:650]
    
        if ret:
            results = model.track(frame_original, conf=confidence, persist=False) # Run YOLOv8 tracking model PERSIST IS FALSE IS NOT USING THE TRACKING
            
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu() #extract bounding boxex of the detected objects
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist() #all track_ids of the objects that are in the current frame
            else:
                track_ids = [] #create empty list for a new track id
            frame_original = results[0].plot() # Visualize the results on the frame_original
            
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                print(track_id)
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > track_time:  # retain 90 tracks for 90 frames
                    track.pop(0)
    
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_original, [points], isClosed=False, color=(green), thickness=2)
                
                # Calculate midpoint
                midpoint_x = int(x)
                midpoint_y = int(y)

                #midpoint 
                x_P, y_P = midpoint_x, midpoint_y
                cv2.circle(frame_original,(x_C,y_C),10,(blue),3)
                cv2.circle(frame_original,(x_P,y_P),6,(red),2)
                
                #relatieve beweging in pixels
                xrev_pixel = x_P - x_C
                yrev_pixel = y_P - y_C
                cv2.line(frame_original, (x_C,y_C), (x_P,y_P), (green),3)
                
                #relatieve beweging in real world
                xrev_wereld = int(xrev_pixel*pix_RW_X)
                yrev_wereld = int(yrev_pixel*pix_RW_Y)
                xrev_wereld = str(xrev_wereld)
                yrev_wereld = str(yrev_wereld)

                print("    i             ", i)
                print("    track_id      ", track_id)
                

                belt_off = str(-5)
                
                if midpoint_x >= virtual_line:
                    # Send coordinates only if they haven't been sent before
                    print("Virtual Line")
                    print("Virtual Line")
                    print("Virtual Line")
                    print("Virtual Line")
                    print("Virtual Line")
                    print("Virtual Line")
                    print("Virtual Line")
                    
                    
                    
                    rev_move = f'{belt_off}'  # Combine DY, DX, and belt_off values
                    client_socket.sendall(rev_move.encode("utf-8"))
                    sent_coordinates = True  # Set the flag to True after sending coordinates#stop_conveyor = True
                    belt_off = 1
                    i +=1
                    
                    belt_off = 0
                    
                    cv2.waitKey(conveyorwait)
                    
                    while sent_coordinates == True:
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        print("while sent_coordinates = True loop!!!!")
                        ret, frame_original = cap.read()

                        if ret:
                            results = model.track(frame_original, conf=confidence, persist=False) # Run YOLOv8 tracking model
                            
                            # Get the boxes and track IDs
                            boxes = results[0].boxes.xywh.cpu()
                            if results[0].boxes.id is not None:
                                track_ids = results[0].boxes.id.int().cpu().tolist()
                        
                            else:
                                track_ids = []
                            frame_original = results[0].plot() # Visualize the results on the frame_original
                          
                            sent_coordinates==False
                            
                            # Plot the tracks
                            for box, track_id in zip(boxes, track_ids):
                                x, y, w, h = box
                                track = track_history[track_id]
                                print(track_id)
                                track.append((float(x), float(y)))  # x, y center point
                                if len(track) > track_time:  # retain 90 tracks for 90 frames
                                    track.pop(0)
                    
                                # Draw the tracking lines
                                #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                #cv2.polylines(frame_original, [points], isClosed=False, color=(green), thickness=2)
                                
                                # Calculate midpoint
                                midpoint_x = int(x)
                                midpoint_y = int(y)
                                
                                #midpoint 
                                #x_C,y_C = 248,128
                                x_P, y_P = midpoint_x, midpoint_y
                                cv2.circle(frame_original,(x_C,y_C),10,(blue),3)
                                #cv2.circle(frame_original,(x_P,y_P),6,(red),2)
                                
                                #relatieve beweging in pixels
                                xrev_pixel = x_P - x_C
                                yrev_pixel = y_P - y_C
                                #cv2.line(frame_original, (x_C,y_C), (x_P,y_P), (green),3)
                                
                                #relatieve beweging in real world
                                xrev_wereld = int(xrev_pixel*pix_RW_X)
                                yrev_wereld = int(yrev_pixel*pix_RW_Y)
                                xrev_wereld = str(xrev_wereld)
                                yrev_wereld = str(yrev_wereld)

                            
                                a = f'{yrev_wereld},{xrev_wereld}'  # Combine DY, DX, and belt_off values
                                client_socket.sendall(a.encode("utf-8"))
                                j +=1
                                last_coordinates = True
                                
                                break
                            break
                        break
                    cv2.waitKey(2)
                    break
                

                
                #Drawing lines and circles
                cv2.arrowedLine(frame_original, (x_C,y_C), (x_P,y_P), (green),3)
                cv2.circle(frame_original, (midpoint_x, midpoint_y), d1, (red), -1)
                cv2.circle(frame_original,(x_C,y_C),d1,(red),-1)
                cv2.circle(frame_original,(x_P,y_P),d1,(red),2)
                cv2.line(frame_original, (virtual_line,0),(virtual_line,480),(pink), 2)
                
                #Put text on frame_original
                textC = f"C: ({x_C}, {y_C})"
                cv2.putText(frame_original, textC, (x_C+30, y_C), font1, size2, (red), thickness2, cv2.LINE_AA)
                textP = f"P: ({x_P}, {y_P})"
                cv2.putText(frame_original, textP, (x_P+30, y_P), font1, size2, (red), thickness2, cv2.LINE_AA)
                
                
                cv2.rectangle(frame_original,(0,0),(240,90), (grey),-1)
                text = f"Midpoint: ({midpoint_x},{midpoint_y})"
                cv2.putText(frame_original, text, (10, 20), font1, size1, (black), thickness1, cv2.LINE_AA)
                text = f"Move: ({yrev_wereld},{xrev_wereld})"
                cv2.putText(frame_original, text, (10, 50), font1, size1, (black), thickness1, cv2.LINE_AA)          
                text = f"ID, i: ({track_id}, {i})"
                cv2.putText(frame_original, text, (10, 80), font1, size1, (black), thickness1, cv2.LINE_AA)
          
                    
              
            # Display theframe
            cv2.imshow("YOLOv8 Tracking", frame_original)
    
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            #Break the loop if the end of the video is reached
            break
    
    
    cap.release()
    cv2.destroyAllWindows()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    message_to_server = "Hello world, this is spyder"           #storing data (str) in a variable
    client_socket.sendall(message_to_server.encode("utf-8"))
    

    while True:
        print("Waiting for command...")
    
        data = client_socket.recv(1024)         #receiving data from server
        server_response = data.decode("utf-8")  #storing data (str) in a variable
        print("Server response:", server_response)

        if server_response == 'exit':
            print('exiting...')
            break
        elif server_response == 'vision_test()':
            vision_test()
