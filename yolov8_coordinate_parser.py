import torch
import cv2
import argparse
import json
import pandas
from ultralytics import YOLO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# yolov8 handler for object detection in images
# handler class

class handler():
    def __init__(self, img_path, model_path):
        self.model = None                       # yolov8 model
        self.img_path = img_path                # path for image
        self.model_path = model_path            # path for model weight

    def load_model(self):
        # loading pytorch yolov8 model for inference
        self.model = YOLO(self.model_path)
        # shifting model to GPU/CPU depending on resource available
        self.model.to(device)

    def img_processing(self):
        print("Starting image processing...")
        print("Inferencing...")
        # reading image
        img = cv2.imread(self.img_path)
        # inferencing
        results = self.model.predict(source=img)
        for res in results:
            # processing for each bounding box in the image
            boxes = res.boxes
            if not boxes:
                self.write_output('', '')
                break
            for box in boxes:
                # bringing the tensor output from GPU to CPU and converting it to numpy array to get the bounding box coordinates
                bbox = box.cpu().numpy().xyxy[0]
                # extracting classname
                classname = self.model.names[int(box.cls)]
                #class probability
                class_probability = box.conf[0].item()
                # converting the numpy array to a dataframe for JSON processing
                data = pandas.DataFrame(bbox)
                # writing the coordinates to a JSON file
                self.write_output(data[0].to_json(orient='records'), classname, class_probability)
        print("JSON file saved!")
        print("Image processed!")

    def write_output(self, data, name, prob):
        # writing JSON format output to a file
        with open("output.json", "a") as outfile:
            # indenting the JSON data according to the YOLOv8 JSON output parameters
            if data == "":
                data = 'No DETECTION'
                result = f'Detections: ' + data + '\n'
            else:
                data = json.dumps(data)
                result = f'Class: {name}\t Detections: ' + data + f'\tClass Probability: {prob}\n'
            outfile.write(result)
        outfile.close()

    def __del__(self):
        # object destructor
        # yolov8 model
        self.model = None
        # path for image
        self.vid_path = None
        # path for model weights
        self.model_path = None
        print("Exiting...")


# main function
if __name__ == '__main__':
    # Arguments from CLI
    '''    
        
        Method to execute code
         
        python yolov8_coordinate_parser.py -img_path (image path)

    '''

    parser = argparse.ArgumentParser(description='I/O and model file paths required.')
    parser.add_argument('-img_path', type=str, dest='img_path', required=True)
    #parser.add_argument('-model_path', type=str, dest='model_path', required=True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # whatever you are timing goes here
    vid_handler = handler(args.img_path, './best.pt')
    vid_handler.load_model()
    vid_handler.img_processing()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:", "%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
