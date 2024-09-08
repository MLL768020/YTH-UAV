import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import RTDETRDetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# class yolov8_heatmap:
#     def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
#         device = torch.device(device)
#         ckpt = torch.load(weight)
#         model_names = ckpt['model'].names
#         csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
#         model = Model(cfg, ch=3, nc=len(model_names)).to(device)
#         csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
#         model.load_state_dict(csd, strict=False)  # load
#         model.eval()
#         print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
#
#         target_layers = [eval(layer)]
#         method = eval(method)
#
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
#         self.__dict__.update(locals())
#
#     def post_process(self, result):
#         logits_ = result[:, 4:]
#         boxes_ = result[:, :4]
#         sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#         return logits_[indices], boxes_[indices], xywh2xyxy(boxes_[indices]).cpu().detach().numpy()
#
#     def draw_detections(self, box, color, name, img):
#         h, w, _ = img.shape
#         box[0] = box[0] * w
#         box[2] = box[2] * w
#         box[1] = box[1] * h
#         box[3] = box[3] * h
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
#                     lineType=cv2.LINE_AA)
#         return img
#
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)
#
#         # img process
#         img = cv2.imread(img_path)
#         img = letterbox(img, auto=False, scaleFill=True)[0]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#
#         # init ActivationsAndGradients
#         grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
#
#         # get ActivationsAndResult
#         result = grads(tensor)
#         activations = grads.activations[0].cpu().detach().numpy()
#
#         # postprocess to yolo output
#         post_result, pre_post_boxes, post_boxes = self.post_process(result[0][0])
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf_threshold:
#                 break
#
#             self.model.zero_grad()
#             # get max probability for this prediction
#             if self.backward_type == 'class' or self.backward_type == 'all':
#                 score = post_result[i].max()
#                 score.backward(retain_graph=True)
#
#             if self.backward_type == 'box' or self.backward_type == 'all':
#                 for j in range(4):
#                     score = pre_post_boxes[i, j]
#                     score.backward(retain_graph=True)
#
#             # process heatmap
#             if self.backward_type == 'class':
#                 gradients = grads.gradients[0]
#             elif self.backward_type == 'box':
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
#             else:
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
#                             grads.gradients[4]
#             b, k, u, v = gradients.size()
#             weights = self.method.get_cam_weights(self.method, None, None, None, activations,
#                                                   gradients.detach().numpy())
#             weights = weights.reshape((b, k, 1, 1))
#             saliency_map = np.sum(weights * activations, axis=1)
#             saliency_map = np.squeeze(np.maximum(saliency_map, 0))
#             saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
#             saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#             if (saliency_map_max - saliency_map_min) == 0:
#                 continue
#             saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#
#             # add heatmap and box to image
#             cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
#             cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i].argmax())],
#                                              f'{self.model_names[int(post_result[i].argmax())]} {float(post_result[i].max()):.2f}',
#                                              cam_image)
#             cam_image = Image.fromarray(cam_image)
#             cam_image.save(f'{save_path}/{i}.png')

class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        self.method = eval(method)
        self.target_layers = target_layers
        self.colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.backward_type = backward_type  # 添加这一行

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return logits_[indices], boxes_[indices], xywh2xyxy(boxes_[indices]).cpu().detach().numpy()

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # img process
        img = cv2.imread(img_path)
        img = letterbox(img, auto=False, scaleFill=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0][0])

        # Process only the first instance
        if post_result.size(0) > 0 and float(post_result[0].max()) >= self.conf_threshold:
            self.model.zero_grad()

            # Get gradients for the first prediction
            score = post_result[0].max()
            score.backward(retain_graph=True)

            # Sum all gradients
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            else:
                gradients = sum(grads.gradients)  # 这里使用 sum() 来对所有梯度求和

            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()

            if (saliency_map_max - saliency_map_min) > 0:
                saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
                cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)

                cam_image = Image.fromarray(cam_image)
                cam_image.save(f'{save_path}/heatmap.png')





def get_params():
    params = {
        'weight': '/home/liyihang/lyhredetr/runs/vis/GIoU/weights/best.pt',
        'cfg': 'ultralytics/cfg/models/rt-detr/llf/vis/yolov8decoder.yaml',
        'device': 'cuda:2',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[15]',
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.3,  # 0.3
        'ratio': 0.5  # 0.5-1.0
    }
    return params


def get_params():
    params = {
        'weight': '/home/cv/lyh/lyhrtdetr2/runs/vis/sppelan_/weights/best.pt',
        'cfg': 'ultralytics/cfg/models/rt-detr/llf/vis/sppelan.yaml',
        'device': 'cuda:2',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[15]',
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.3,  # 0.3
        'ratio': 0.5  # 0.5-1.0
    }
    return params



if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())

    model(r'/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/detect/0000001_05999_d_0000011.jpg', 'uav111')
    model(r'/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/detect/0000001_07999_d_0000012.jpg', 'uav112')
    model(r'/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/detect/0000115_00796_d_0000081.jpg', 'uav113')
    model(r'/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/detect/0000295_01800_d_0000030.jpg', 'uav114')
    model(r'/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/detect/0000296_01001_d_0000040.jpg', 'uav115')