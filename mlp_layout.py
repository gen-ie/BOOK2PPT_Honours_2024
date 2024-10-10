# Object Placer 
import torch as T
import torch.nn as nn

device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Define the MLP for layout prediction
class MLPlayout(nn.Module):
    def __init__(self):
        super(MLPlayout, self).__init__()
        self.fc1 = nn.Linear(1024, 256)  # Adjust in_features based on input size
        self.fc2 = nn.Linear(256, 64)  
        self.fc3 = nn.Linear(64, 4, bias=False) # Output: [x, y, width, height]

    def forward(self, x):
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        x = T.relu(self.fc3(x))
        return x

# Alternative loss function for Object Placer
class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def convert_to_xyxy(self, boxes, slide_size):
        slide_width, slide_height = slide_size

        # Convert to original size (pixels)
        left = boxes[:, 0] * slide_width
        top = boxes[:, 1] * slide_height
        width = boxes[:, 2] * slide_width
        height = boxes[:, 3] * slide_height
        
        # Get x1, y1, x2, y2, then normalise
        x1 = (left) / slide_width
        y1 = (top) / slide_height
        x2 = (left + width) / slide_width
        y2 = (top + height) / slide_height

        return T.stack([x1, y1, x2, y2], dim=1)

    def forward(self, preds, targets, slide_size):
        # Extract predictions and targets
        pred_boxes = self.convert_to_xyxy(preds, slide_size)
        true_boxes = self.convert_to_xyxy(targets, slide_size)

        # Calculate the coordinates of the intersection rectangle
        x1 = T.max(pred_boxes[:, 0], true_boxes[:, 0])
        y1 = T.max(pred_boxes[:, 1], true_boxes[:, 1])
        x2 = T.min(pred_boxes[:, 2], true_boxes[:, 2])
        y2 = T.min(pred_boxes[:, 3], true_boxes[:, 3])

        # Calculate intersection area
        inter_area = T.max(x2 - x1, T.tensor(0.0)) * T.max(y2 - y1, T.tensor(0.0))

        # Calculate areas of both boxes
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

        # Calculate union area
        union_area = pred_area + true_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)

        # Calculate the center points
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        true_center_x = (true_boxes[:, 0] + true_boxes[:, 2]) / 2
        true_center_y = (true_boxes[:, 1] + true_boxes[:, 3]) / 2

        # Calculate the center distance
        center_distance = (pred_center_x - true_center_x) ** 2 + (pred_center_y - true_center_y) ** 2

        # Calculate the diagonal length of the smallest enclosing box
        enc_x1 = T.min(pred_boxes[:, 0], true_boxes[:, 0])
        enc_y1 = T.min(pred_boxes[:, 1], true_boxes[:, 1])
        enc_x2 = T.max(pred_boxes[:, 2], true_boxes[:, 2])
        enc_y2 = T.max(pred_boxes[:, 3], true_boxes[:, 3])
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

        # Calculate the aspect ratio consistency
        aspect_ratio = T.exp(-1 * (enc_diag - center_distance) / (enc_diag + 1e-6))

        # CIoU loss
        ciou_loss = 1 - iou + aspect_ratio

        return ciou_loss.mean()