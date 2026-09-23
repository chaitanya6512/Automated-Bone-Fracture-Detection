import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import threading
from tkinter import filedialog
from pathlib import Path
import customtkinter as ctk
from PIL import Image
from core.gradcam import generate_gradcam
from core.predictions import predict
from core.pdf_report import generate_pdf_report

# global variables

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"


#def predict_gui(self):
#    thread = threading.Thread(target=self._run_prediction)
#    thread.daemon = True
#    thread.start()

#def _run_prediction(self):
#    # all the prediction logic here
#    # use self.after() to update GUI from main thread
#    self.after(0, self._update_ui, results)

from core.preprocessing import read_image_file

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.filename = None

        self.title("Automated Bone Fracture Detection (Grad-CAM++)")
        self.geometry("950x700")
        self.minsize(900, 650)

        self.head_frame = ctk.CTkFrame(master=self)
        self.head_frame.pack(pady=20, padx=60, fill="x")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(
            pady=20,
            padx=20,
            fill="both",
            expand=True
        )

        self.tabview.add("Upload")
        self.tabview.add("Prediction")
        self.tabview.add("Evaluation")
        self.tabview.add("About")

        self.upload_tab = ctk.CTkScrollableFrame(
            self.tabview.tab("Upload")
        )

        self.upload_tab.pack(
            fill="both",
            expand=True
        )

        self.prediction_tab = ctk.CTkScrollableFrame(
            self.tabview.tab("Prediction")
        )

        self.prediction_tab.pack(
            fill="both",
            expand=True
        )

        self.evaluation_tab = ctk.CTkScrollableFrame(
            self.tabview.tab("Evaluation")
        )

        self.evaluation_tab.pack(
            fill="both",
            expand=True
        )

        self.about_tab = ctk.CTkScrollableFrame(
            self.tabview.tab("About")
        )

        self.about_tab.pack(
            fill="both",
            expand=True
        )

        # =========================================
        # PREDICTION TAB
        # =========================================

        self.prediction_title = ctk.CTkLabel(
            master=self.prediction_tab,
            text="Prediction Results",
            font=("Arial", 26, "bold")
        )

        self.prediction_title.pack(pady=20)

        self.res1_label = ctk.CTkLabel(
            master=self.prediction_tab,
            text="",
            font=("Arial", 18)
        )

        self.res2_label = ctk.CTkLabel(
            master=self.prediction_tab,
            text="",
            font=("Arial", 18, "bold")
        )

        self.gradcam_title = ctk.CTkLabel(
            master=self.prediction_tab,
            text="Diagnostic Visualization (Grad-CAM++)",
            font=("Arial", 22, "bold")
        )

        self.gradcam_label = ctk.CTkLabel(
            master=self.prediction_tab,
            text=""
        )

        self.pdf_btn = ctk.CTkButton(
            master=self.prediction_tab,
            text="Export PDF Report",
            command=self.generate_pdf
        )

        self.save_label = ctk.CTkLabel(
            master=self.prediction_tab,
            text=""
        )

        self.gradcam_title.pack_forget()
        self.gradcam_label.pack_forget()
        self.pdf_btn.pack_forget()
        self.save_label.pack_forget()

        # =========================================
        # EVALUATION TAB
        # =========================================

        self.eval_title = ctk.CTkLabel(
            master=self.evaluation_tab,
            text="Model Evaluation",
            font=("Arial", 24, "bold")
        )

        self.eval_title.pack(pady=20)

        # -----------------------------------------
        # BODY PARTS ROC
        # -----------------------------------------

        body_roc_path = Path(
            "plots/Evaluation/BodyParts_ROC_Curve.png"
        )

        if body_roc_path.exists():

            body_roc_image = ctk.CTkImage(
                light_image=Image.open(body_roc_path),
                dark_image=Image.open(body_roc_path),
                size=(450, 450)
            )

            self.body_roc_label = ctk.CTkLabel(
                master=self.evaluation_tab,
                image=body_roc_image,
                text=""
            )

            self.body_roc_label.pack(pady=10)

        # -----------------------------------------
        # BODY PARTS CONFUSION MATRIX
        # -----------------------------------------

        body_cm_path = Path(
            "plots/Evaluation/BodyParts_ConfusionMatrix.png"
        )

        if body_cm_path.exists():

            body_cm_image = ctk.CTkImage(
                light_image=Image.open(body_cm_path),
                dark_image=Image.open(body_cm_path),
                size=(450, 450)
            )

            self.body_cm_label = ctk.CTkLabel(
                master=self.evaluation_tab,
                image=body_cm_image,
                text=""
            )

            self.body_cm_label.pack(pady=10)

        # -----------------------------------------
        # ELBOW ROC
        # -----------------------------------------

        elbow_roc_path = Path(
            "plots/Evaluation/Elbow_ROC_Curve.png"
        )

        if elbow_roc_path.exists():

            elbow_roc_image = ctk.CTkImage(
                light_image=Image.open(elbow_roc_path),
                dark_image=Image.open(elbow_roc_path),
                size=(450, 450)
            )

            self.elbow_roc_label = ctk.CTkLabel(
                master=self.evaluation_tab,
                image=elbow_roc_image,
                text=""
            )

            self.elbow_roc_label.pack(pady=10)

        # -----------------------------------------
        # HAND ROC
        # -----------------------------------------

        hand_roc_path = Path(
            "plots/Evaluation/Hand_ROC_Curve.png"
        )

        if hand_roc_path.exists():

            hand_roc_image = ctk.CTkImage(
                light_image=Image.open(hand_roc_path),
                dark_image=Image.open(hand_roc_path),
                size=(450, 450)
            )

            self.hand_roc_label = ctk.CTkLabel(
                master=self.evaluation_tab,
                image=hand_roc_image,
                text=""
            )

            self.hand_roc_label.pack(pady=10)

        # -----------------------------------------
        # SHOULDER ROC
        # -----------------------------------------

        shoulder_roc_path = Path(
            "plots/Evaluation/Shoulder_ROC_Curve.png"
        )

        if shoulder_roc_path.exists():

            shoulder_roc_image = ctk.CTkImage(
                light_image=Image.open(shoulder_roc_path),
                dark_image=Image.open(shoulder_roc_path),
                size=(450, 450)
            )

            self.shoulder_roc_label = ctk.CTkLabel(
                master=self.evaluation_tab,
                image=shoulder_roc_image,
                text=""
            )

            self.shoulder_roc_label.pack(pady=10)

        
        # =========================================
        # ABOUT TAB
        # =========================================

        about_text = """
        Automated Bone Fracture Detection System

        This system uses Deep Learning (DenseNet121) and Grad-CAM++
        to detect fractures from X-ray images.

        Supported Body Parts:
        • Elbow
        • Hand
        • Shoulder

        Features:
        • Automatic body-part classification
        • Fracture detection
        • Grad-CAM++ spatial visualization
        • DICOM (.dcm) & Standard Image Support
        • PDF diagnostic reports
        • ROC & evaluation analysis

        Developed using:
        Python, TensorFlow, CustomTkinter, OpenCV, pydicom
        """

        self.about_label = ctk.CTkLabel(
            master=self.about_tab,
            text=about_text,
            justify="left",
            font=("Arial", 18)
        )

        self.about_label.pack(
            pady=30,
            padx=30,
            anchor="w"
        )
        
        

        # =========================================
        # UPLOAD TAB
        # =========================================
        self.head_label = ctk.CTkLabel(master=self.head_frame, text="Automated Bone Fracture Detection",
                                       font=(ctk.CTkFont("Roboto"), 26))
        self.head_label.pack(pady=20, padx=10, anchor="nw", side="left")
        img1 = ctk.CTkImage(Image.open(IMAGES_DIR / "info.jpeg"))

        self.info_button = ctk.CTkButton(master=self.head_frame, text="", image=img1, command=self.open_image_window,
                                       width=40, height=40)

        self.info_button.pack(pady=10, padx=10, anchor="nw", side="right")

        self.info_label = ctk.CTkLabel(master=self.upload_tab,
                                       text="Automated bone fracture detection system. Supports PNG, JPG, JPEG, and DICOM (.dcm) X-rays.",
                                       wraplength=300, font=(ctk.CTkFont("Roboto"), 18))
        self.info_label.pack(pady=10, padx=10)

        self.upload_btn = ctk.CTkButton(master=self.upload_tab, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=0, padx=1)

        img = Image.open(IMAGES_DIR / "Upload.jpeg")

        self.default_image = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(256, 256)
        )

        self.img_label = ctk.CTkLabel(
            master=self.upload_tab,
            text="",
            image=self.default_image
        )
        self.img_label.pack(pady=1, padx=10)


        self.predict_btn = ctk.CTkButton(master=self.upload_tab, text="Predict", command=self.start_prediction_thread)
        self.predict_btn.pack(pady=0, padx=1)

 
    def upload_image(self):
        try:
            f_types = [
                ("Medical X-Rays / DICOM", "*.png;*.jpg;*.jpeg;*.dcm;*.dicom"),
                ("All Files", "*.*")
            ]
            self.filename = filedialog.askopenfilename(
                filetypes=f_types,
                initialdir=os.path.expanduser("~/Pictures")
            )

            if not self.filename:
                return

            self.save_label.configure(text="")
            self.res2_label.configure(text="")
            self.res1_label.configure(text="")
            self.img_label.configure(
                text="",
                image=self.default_image
            )

            rgb_array = read_image_file(self.filename)
            img = Image.fromarray(rgb_array)

            self.uploaded_ctk_image = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(256, 256)
            )

            self.img_label.configure(
                image=self.uploaded_ctk_image,
                text=""
            )

            self.save_label.pack_forget()

        except Exception as e:
            print("Error loading image:", e)

    
    def start_prediction_thread(self):

        self.predict_btn.configure(
            state="disabled",
            text="Predicting..."
        )

        thread = threading.Thread(
            target=self.predict_gui
        )

        thread.daemon = True

        thread.start()


    def predict_gui(self):

        try:

            if not self.filename:
                print("No image selected.")
                return

            # -----------------------------------------
            # Predict BODY PART
            # -----------------------------------------
            bone_type_result, bone_conf = predict(
                self.filename,
                "Parts",
                return_confidence=True
            )

            # -----------------------------------------
            # Predict FRACTURE & UNCERTAINTY (1c)
            # -----------------------------------------
            fracture_result, fracture_conf, probs, is_uncertain = predict(
                self.filename,
                bone_type_result,
                return_uncertainty=True
            )

            self.tabview.set("Prediction")

            self.after(
                0,
                lambda: self.predict_btn.configure(
                    state="normal",
                    text="Predict"
                )
            )

            # -----------------------------------------
            # Update GUI labels
            # -----------------------------------------
            self.res1_label.configure(
                text=f"Detected Body Part: {bone_type_result} ({bone_conf:.2f}%)"
            )

            result_str = f"Fracture Detection: {fracture_result.upper()} ({fracture_conf:.2f}%)"
            if is_uncertain:
                result_str += "\n⚠️ CLINICAL WARNING: Prediction is UNCERTAIN (<65% confidence). Radiologist review required."

            self.res2_label.configure(
                text=result_str
            )

            self.res1_label.pack(pady=10)
            self.res2_label.pack(pady=10)

            # -----------------------------------------
            # Generate GradCAM++
            # -----------------------------------------
            gradcam_path = generate_gradcam(
                self.filename,
                bone_type_result
            )

            grad_img = Image.open(gradcam_path)
            grad_img = grad_img.resize((300, 300))

            grad_img = ctk.CTkImage(
                light_image=grad_img,
                dark_image=grad_img,
                size=(300, 300)
            )

            self.gradcam_label.configure(
                image=grad_img,
                text=""
            )
            self.gradcam_title.pack(pady=20)
            self.gradcam_label.pack(pady=10)
            self.pdf_btn.pack(pady=20)
            self.gradcam_label.pack(pady=5)

            # -----------------------------------------
            # Store results for PDF
            # -----------------------------------------
            self.body_part = bone_type_result
            self.body_conf = bone_conf
            self.fracture_result = fracture_result
            self.fracture_conf = fracture_conf
            self.is_uncertain = is_uncertain
            self.gradcam_path = gradcam_path

            self.pdf_btn.pack(pady=10)

        except Exception as e:

            print("Prediction Error:", e)

            self.after(
                0,
                lambda: self.predict_btn.configure(
                    state="normal",
                    text="Predict"
                )
            )

    def generate_pdf(self):
        try:
            save_path = filedialog.asksaveasfilename(
                parent=self,
                initialdir=os.path.expanduser("~/Documents"),
                title="Save PDF Report",
                defaultextension=".pdf",
                filetypes=[("PDF File", "*.pdf")]
            )

            if not save_path:
                return

            generate_pdf_report(
                save_path=save_path,
                original_image_path=self.filename,
                gradcam_image_path=self.gradcam_path,
                body_part=self.body_part,
                body_conf=self.body_conf,
                fracture_result=self.fracture_result,
                fracture_conf=self.fracture_conf,
                is_uncertain=getattr(self, 'is_uncertain', False)
            )

            self.save_label.configure(
                text="PDF Report Saved Successfully!",
                text_color="GREEN",
                font=(ctk.CTkFont("Roboto"), 16)
            )

            self.save_label.pack(pady=5)

        except Exception as e:
            print("PDF generation error:", e)

        
    def open_image_window(self):
        try:
            im = Image.open(IMAGES_DIR / "rules.jpeg")
            im = im.resize((700, 700))
            im.show()
        except Exception as e:
            print("Error opening rules image:", e)


if __name__ == "__main__":
    app = App()
    app.mainloop()
