# from torch.xpu import device

from ultralytics import YOLO

def classification(tasks='cls', data_path, ):
    write_yaml('temp', 'nihao')
    a()
    model = YOLO("yolo11n.yaml")
    results = model.train(data="coco8.yaml", epochs=8, device='cpu')
    results = model.val(data='coco8.yaml')
    results = model.predict("./bus.jpg", save=True, show=True)
    return results

if __name__ == '__main__':
    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO("our.pt")
    # Create a new YOLO model from scratch
    print(classification(tasks='cls', data_path='./temp/nihao.yaml'))

    model = YOLO("our.pt")
    results = model.val(data='coco8.yaml')
    results = model.predict("./bus.jpg", save=True, show=True)
