from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('inputvideo/08fd33_4.mp4', save=True)


