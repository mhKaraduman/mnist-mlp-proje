import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import os

class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def center_by_mass(img_28x28):
    """MNIST standardı: Rakamı piksellerin kütle merkezine göre tam ortaya (14,14) hizalar."""
    moments = cv2.moments(img_28x28)
    if moments["m00"] == 0:
        return img_28x28
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    shift_x = int(round(14.0 - cx))
    shift_y = int(round(14.0 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img_28x28, M, (28, 28))

def process_and_predict_student_id(image_path, model_path='mnist_mlp_model.pth'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Hata: '{image_path}' okunamadı!")
        return

    # 1. Sadece en üst ve en alttaki beyaz ekran görüntüsü şeritlerini kırp (sağ ve sola dokunma, 4 rakamı gitmesin)
    h_img, w_img = img.shape
    img = img[int(h_img * 0.14):int(h_img * 0.92), 2:w_img-2]
    h_img, w_img = img.shape

    # 2. Daha hassas eşikleme (C=7 sayesinde silik kalem izleri asla kaybolmaz)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 61, 7)

    # 3. Küçük gürültü noktalarını temizle, sonra kalem çizgilerini birleştirip kalınlaştır
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Çizgileri MNIST gibi dolgun ve parlak yap
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thickened = cv2.dilate(closed, kernel_dilate, iterations=2)

    # 4. Rakam kutularını bul
    contours, _ = cv2.findContours(thickened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > h_img * 0.22 and w > 4:
            raw_boxes.append([x, y, w, h])

    raw_boxes = sorted(raw_boxes, key=lambda b: b[0])

    # Yatayda birbirine çok yakın/kopuk parçaları (ör. 5'in üst çizgisi veya 4'ün parçası) tek kutuda birleştir
    digit_bboxes = []
    for box in raw_boxes:
        if not digit_bboxes:
            digit_bboxes.append(box)
        else:
            px, py, pw, ph = digit_bboxes[-1]
            cx, cy, cw, ch = box
            # Kutular yatayda iç içe geçmişse birleştir
            overlap = min(px + pw, cx + cw) - max(px, cx)
            if overlap > -3:
                nx = min(px, cx)
                ny = min(py, cy)
                nw = max(px + pw, cx + cw) - nx
                nh = max(py + ph, cy + ch) - ny
                digit_bboxes[-1] = [nx, ny, nw, nh]
            else:
                digit_bboxes.append(box)

    print(f"Görselde toplam {len(digit_bboxes)} adet rakam tespit edildi.")

    os.makedirs("mnist_rakamlarim", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    processed_digits = []
    predictions = []
    confidences = []

    # 5. Her rakamı 28x28 MNIST formatına çevir
    for i, (x, y, w, h) in enumerate(digit_bboxes):
        roi = thickened[y:y+h, x:x+w]

        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Küçültme sonrası silikleşen pikselleri tekrar parlak beyaz yap (Kontrast artırma)
        resized = np.clip(resized * 1.5, 0, 255).astype(np.uint8)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        # Kütle merkezine göre ortala
        canvas = center_by_mass(canvas)

        cv2.imwrite(f"mnist_rakamlarim/rakam_{i+1}.png", canvas)
        processed_digits.append(canvas)

        tensor_img = transform(canvas).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor_img)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predictions.append(pred.item())
            confidences.append(conf.item() * 100)

    # 6. Sonuçları Görselleştir
    if len(processed_digits) > 0:
        plt.figure(figsize=(15, 3))
        for i, digit_img in enumerate(processed_digits):
            plt.subplot(1, len(processed_digits), i + 1)
            plt.imshow(digit_img, cmap='gray')
            plt.title(f"Tahmin: {predictions[i]}\n(%{confidences[i]:.0f})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("okul_numarasi_mnist_sonuc.png", dpi=150)
        print("Tahmin Edilen Okul Numarası:", "".join(map(str, predictions)))
        print("Sonuç görseli 'okul_numarasi_mnist_sonuc.png' olarak güncellendi!")

if __name__ == "__main__":
    for filename in ['student_id.jpg', 'student_id.png', '.student_id.jpg']:
        if os.path.exists(filename):
            print(f"Kullanılan görsel: {filename}")
            process_and_predict_student_id(filename)
            break