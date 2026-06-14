import os
import cv2
from fpdf import FPDF

# İşlenen test resimleri
TEST_RESULTS_DIR = "test_results"
PDF_PATH = "Test_Report.pdf"

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)

for class_name in os.listdir(TEST_RESULTS_DIR):
    class_path = os.path.join(TEST_RESULTS_DIR, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        # Resim boyutunu PDF'e sığacak şekilde ayarla
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width, _ = img.shape
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{class_name} - {img_name}", ln=True)
        # OpenCV BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp_img_path = "tmp.jpg"
        cv2.imwrite(tmp_img_path, img_rgb)
        pdf.image(tmp_img_path, x=10, y=25, w=pdf.w-20)
        os.remove(tmp_img_path)

pdf.output(PDF_PATH)
print(f"PDF rapor oluşturuldu: {PDF_PATH}")
