import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
from tkinter import filedialog, Tk

# ================= MODELS =================
person_model = YOLO("yolov8n.pt")
weapon_model = YOLO("best.pt")

cap = None
threshold_value = 20
trackbar_max = 100

buttons = {}
pressed_button = None

heatmap = None

time_data = []
count_data = []
start_time = None
last_graph_update = None
current_second = 0
second_counts = []

fig, ax = plt.subplots(figsize=(12, 4), dpi=80)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

graph_img = None
last_weapon_warning_time = None
warning_duration = 1.0
weapon_confidence = 0.55
knife_confidence = 0.40
person_confidence = 0.3

# ================= UI =================
def draw_button(frame, x, y, w, h, text, bid):
    color = (0,200,0) if pressed_button != bid else (0,150,0)
    cv2.rectangle(frame,(x,y),(x+w,y+h),color,-1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,0.6,1)[0]
    tx = x + (w-size[0])//2
    ty = y + (h+size[1])//2
    cv2.putText(frame,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    buttons[bid]=(x,y,x+w,y+h)

def draw_toolbar(frame,status):
    bar = np.ones((100,frame.shape[1],3),dtype=np.uint8)*40
    buttons.clear()

    draw_button(bar,10,10,120,35,"WEBCAM","cam")
    draw_button(bar,140,10,150,35,"UPLOAD VIDEO","vid")

    cv2.putText(bar,f"Source: {status}",(320,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,200),1)

    cv2.putText(bar,f"Threshold: {threshold_value}",(620,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    return np.vstack((bar,frame))

# ================= MAIN =================
def run():
    global cap, heatmap, start_time, current_second
    global second_counts, last_graph_update, graph_img, pressed_button
    global time_data, count_data, threshold_value
    global last_weapon_warning_time
    knife_confirmation_streak = 0

    cv2.namedWindow("Crowd Monitoring System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Monitoring System", 1280, 800)
    cv2.createTrackbar("Threshold","Crowd Monitoring System", int(threshold_value), trackbar_max, lambda x:None)

    start_time=time.time()
    last_graph_update=start_time

    def mouse(event,x,y,flags,param):
        global cap, pressed_button

        if event==cv2.EVENT_LBUTTONDOWN:
            for k,(x1,y1,x2,y2) in buttons.items():
                if x1<=x<=x2 and y1<=y<=y2:
                    pressed_button=k

        elif event==cv2.EVENT_LBUTTONUP:
            for k,(x1,y1,x2,y2) in buttons.items():
                if x1<=x<=x2 and y1<=y<=y2:
                    if k=="cam":
                        if cap: cap.release()
                        cap=cv2.VideoCapture(0)
                        if not cap.isOpened():
                            print("Webcam error")
                            cap=None

                    elif k=="vid":
                        root=Tk(); root.withdraw()
                        path=filedialog.askopenfilename()
                        root.destroy()
                        if path:
                            if cap: cap.release()
                            cap=cv2.VideoCapture(path)

            pressed_button=None

    cv2.setMouseCallback("Crowd Monitoring System",mouse)

    while True:
        threshold_value = cv2.getTrackbarPos("Threshold","Crowd Monitoring System")
        if threshold_value < 1:
            threshold_value = 1

        if cap is None or not cap.isOpened():
            status="None"
            frame = np.ones((600, 1000, 3), dtype=np.uint8) * 50
            cv2.putText(frame, "No video source selected",
                        (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 165, 255), 2)
            
            cv2.putText(frame, "Click WEBCAM or UPLOAD VIDEO",
                        (200, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 165, 255), 2)
            
            combined = frame
            display = combined

            if graph_img is not None:
                if graph_img.shape[1] != combined.shape[1]:
                    graph_img = cv2.resize(graph_img, (combined.shape[1], graph_img.shape[0]))

                graph_height = 300
                graph_img = cv2.resize(graph_img, (combined.shape[1], graph_height))
                cv2.line(combined,
                        (0, combined.shape[0]-1),
                        (combined.shape[1], combined.shape[0]-1),
                        (255,255,255), 2)
                display = np.vstack((combined, graph_img))
        else:
            ret,frame = cap.read()
            if not ret or frame is None:
                if cap:
                    cap.release()
                cap = None
                continue

            h,w = frame.shape[:2]

            if heatmap is None:
                heatmap = np.zeros((h,w), dtype=np.float32)

            status="Webcam"

            # ===== PERSON / KNIFE =====
            pres = person_model.predict(frame, conf=min(person_confidence, knife_confidence), imgsz=1280)
            count = 0
            persons = []
            knives = []

            for r in pres:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cls = int(b.cls[0])
                    label = str(person_model.names.get(cls, cls)).strip().lower()

                    if label == "person":
                        persons.append((x1,y1,x2,y2))
                        count += 1
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cx,cy = (x1+x2)//2, (y1+y2)//2
                        cx = int(np.clip(cx, 0, w - 1))
                        cy = int(np.clip(cy, 0, h - 1))
                        heatmap[cy, cx] += 1
                    elif label == "knife":
                        knives.append((x1,y1,x2,y2))
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)
                        cv2.putText(frame, f"{label} {float(b.conf[0]):.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            # ===== PISTOL =====
            wres = weapon_model.predict(frame, conf=weapon_confidence, imgsz=1280)
            pistols = []

            for r in wres:
                for b in r.boxes:
                    conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                    cls = int(b.cls[0])
                    label = str(weapon_model.names.get(cls, cls)).strip().lower()

                    if label != "pistol":
                        continue

                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    pistols.append((x1,y1,x2,y2))
                    box_color = (0, 0, 255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)
                    cv2.putText(frame,f"{label} {conf:.2f}",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,box_color,2)

            # ===== ALERT =====
            threats = knives + pistols
            dangerous_person_with_weapon = False
            for p in persons:
                for wbox in threats:
                    if not (p[2] < wbox[0] or wbox[2] < p[0] or p[3] < wbox[1] or wbox[3] < p[1]):
                        dangerous_person_with_weapon = True

            threshold_warning = count > threshold_value
            current_time = time.time()

            if dangerous_person_with_weapon:
                knife_confirmation_streak = min(5, knife_confirmation_streak + 1)
            else:
                knife_confirmation_streak = max(0, knife_confirmation_streak - 1)

            weapon_confirmed = dangerous_person_with_weapon and knife_confirmation_streak >= 3
            if weapon_confirmed:
                last_weapon_warning_time = current_time

            weapon_warning = False
            if last_weapon_warning_time is not None:
                weapon_warning = (current_time - last_weapon_warning_time) < warning_duration

            warning_texts = []
            if weapon_warning:
                cv2.putText(frame, "ALERT: PERSON WITH WEAPON!",
                            (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                warning_texts.append("WARNING: Weapon confirmed!")
            if threshold_warning:
                warning_texts.append("WARNING: Crowd threshold exceeded!")

            for idx, text in enumerate(warning_texts):
                cv2.putText(frame, text, (50, 150 + idx * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if weapon_warning or threshold_warning:
                cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 4)

            # ===== GRAPH DATA =====
            t = current_time - start_time
            second_counts.append(count)
            if int(t) > current_second:
                current_second = int(t)
                if second_counts:
                    time_data.append(current_second)
                    count_data.append(min(second_counts))
                    second_counts = []
                if len(time_data) > 50:
                    time_data = time_data[-50:]
                    count_data = count_data[-50:]

            # ===== GRAPH UPDATE =====
            if time.time() - last_graph_update >= 2 and len(time_data) > 0:
                last_graph_update = time.time()
                ax.clear()
                ax.set_facecolor('black')
                ax.plot(time_data, count_data, color='#00FF00', linewidth=3)

                # labels
                ax.set_xlabel("Time (seconds)", color='white')
                ax.set_ylabel("Persons detected", color='white')

                # styling (important for good look)
                ax.grid(True, alpha=0.5, color='#00FF00', linewidth=0.5)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('#00FF00')
                ax.spines['left'].set_color('#00FF00')
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')

                # render to image
                canvas = FigureCanvasAgg(fig)
                canvas.draw()

                buf = canvas.buffer_rgba()
                size = canvas.get_width_height()

                graph_img = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], 4)
                graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)


            # ===== HEATMAP =====
            heatmap*=0.95
            hm=cv2.GaussianBlur(heatmap,(51,51),0)
            hm=cv2.normalize(hm,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            hm=cv2.applyColorMap(hm,cv2.COLORMAP_HOT)
            hm=cv2.resize(hm,(w,h))

            combined=np.hstack((frame,hm))
            cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 1)

            cv2.putText(combined,f"Person Count: {count}",(20,40),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

            if graph_img is not None:
                if graph_img.shape[1] != combined.shape[1]:
                    graph_img = cv2.resize(graph_img, (combined.shape[1], graph_img.shape[0]))
                top_h = combined.shape[0]
                display = np.vstack((combined,graph_img))
                cv2.line(display, (0, top_h), (display.shape[1], top_h), (255, 255, 255), 1)
            else:
                display = combined

        display = draw_toolbar(display,status)
        cv2.imshow("Crowd Monitoring System",display)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run()
