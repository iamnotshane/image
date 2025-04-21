import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import torch
import cv2
import numpy as np
import base64
import sys

# Add YOLOv5 path if needed
YOLOV5_PATH = "yolov5"
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Load YOLOv5 model
model = torch.hub.load(YOLOV5_PATH, 'yolov5s', source='local', force_reload=False)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, radius=15, bg="#10a37f", fg="white", 
                 font=("Helvetica", 12), padx=20, pady=8, **kwargs):
        super().__init__(parent, bd=0, highlightthickness=0, relief='ridge', **kwargs)
        self.config(bg=parent["bg"])
        
        self.bg = bg
        self.command = command
        
        text_font = self.create_text(0, 0, text=text, font=font, fill=fg)
        text_bbox = self.bbox(text_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        self.delete(text_font)
        
        width = text_width + padx * 2
        height = text_height + pady * 2
        
        self.create_rounded_rect(0, 0, width, height, radius, fill=bg, outline="")
        self.create_text(width // 2, height // 2, text=text, font=font, fill=fg)
        self.config(width=width, height=height)
        
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def _on_press(self, event):
        self.itemconfig(1, fill=self._darken_color(self.bg))
    
    def _on_release(self, event):
        self.itemconfig(1, fill=self.bg)
        if self.command:
            self.command()
    
    def _on_enter(self, event):
        self.itemconfig(1, fill=self._lighten_color(self.bg))
    
    def _on_leave(self, event):
        self.itemconfig(1, fill=self.bg)
    
    def _darken_color(self, hex_color, factor=0.8):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _lighten_color(self, hex_color, factor=1.1):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def configure(self, **kwargs):
        if "state" in kwargs:
            if kwargs["state"] == "disabled":
                self.itemconfig(1, fill="#cccccc")
            else:
                self.itemconfig(1, fill=self.bg)
        super().configure(**kwargs)

class ImageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recognition")
        self.root.geometry("800x700")
        
        self.theme = {
            "bg_light": "#f7f7f8",
            "bg_dark": "#343541",
            "primary": "#10a37f",
            "secondary": "#444654",
            "accent": "#6e6e80",
            "text_light": "#ffffff",
            "text_dark": "#202123"
        }
        
        self.root.configure(bg=self.theme["bg_light"])
        
        self.image_path = None
        self.original_image = None
        
        self.create_widgets()
    
    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg=self.theme["bg_light"], height=80)
        title_frame.pack(fill=tk.X, pady=(20, 0))
        
        title_label = tk.Label(title_frame, text="Image Recognition", 
                              font=("Helvetica", 28, "bold"), 
                              fg=self.theme["primary"],
                              bg=self.theme["bg_light"])
        title_label.pack(pady=10)
        
        subtitle = tk.Label(self.root, text="Upload an image and let AI identify it", 
                           font=("Helvetica", 14), 
                           fg=self.theme["accent"],
                           bg=self.theme["bg_light"])
        subtitle.pack(pady=(0, 20))
        
        content_frame = tk.Frame(self.root, bg=self.theme["bg_light"])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        self.image_frame = tk.Frame(content_frame, bg="white", 
                                   width=500, height=400,
                                   highlightbackground="#e5e5e5",
                                   highlightthickness=1)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        buttons_frame = tk.Frame(content_frame, bg=self.theme["bg_light"])
        buttons_frame.pack(pady=20)
        
        self.upload_btn = RoundedButton(buttons_frame, text="Choose Image", 
                                      command=self.upload_image,
                                      bg=self.theme["primary"], 
                                      fg=self.theme["text_light"],
                                      font=("Helvetica", 12, "bold"))
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.analyze_btn = RoundedButton(buttons_frame, text="Analyze Image", 
                                       command=self.analyze_image,
                                       bg=self.theme["secondary"], 
                                       fg=self.theme["text_light"],
                                       font=("Helvetica", 12, "bold"))
        self.analyze_btn.pack(side=tk.LEFT, padx=10)
        self.analyze_btn.configure(state="disabled")
        
        self.results_frame = tk.Frame(content_frame, bg="white", 
                                     highlightbackground="#e5e5e5",
                                     highlightthickness=1,
                                     padx=20, pady=20)
        self.results_frame.pack(pady=20, fill=tk.X)
        
        results_title = tk.Label(self.results_frame, text="Analysis Results", 
                               font=("Helvetica", 14, "bold"),
                               fg=self.theme["primary"],
                               bg="white")
        results_title.pack(anchor="w")
        
        tk.Frame(self.results_frame, height=1, bg="#e5e5e5").pack(fill=tk.X, pady=10)
        
        self.results_label = tk.Label(self.results_frame, 
                                     text="Upload an image to begin analysis",
                                     font=("Helvetica", 12),
                                     bg="white",
                                     fg=self.theme["accent"],
                                     wraplength=700,
                                     justify=tk.LEFT)
        self.results_label.pack(anchor="w")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var,
                                  font=("Helvetica", 10),
                                  bg="#f0f0f0", fg=self.theme["accent"],
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                  padx=10, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        file_types = [("Image files", "*.jpg *.jpeg *.png")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                self.image_path = file_path
                self.display_image(file_path)
                self.analyze_btn.configure(state="normal")
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                self.results_label.config(text="Click 'Analyze Image' to identify the image")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def display_image(self, file_path):
        self.original_image = Image.open(file_path)
        width, height = self.original_image.size
        max_width = 480
        max_height = 380
        
        if width > height:
            new_width = min(width, max_width)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(height, max_height)
            new_width = int(width * (new_height / height))
            
        resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image
    
    def analyze_image(self):
        if not self.image_path:
            messagebox.showinfo("Info", "Please upload an image first.")
            return
        
        self.status_var.set("Analyzing image...")
        self.results_label.config(text="Processing... Please wait")
        self.root.update()
        
        try:
            results = model(self.image_path)
            predictions = results.pandas().xyxy[0]
            
            if predictions.empty:
                self.results_label.config(text="No objects detected in the image.")
                self.status_var.set("No objects detected")
                return
            
            result_text = "✓ Detected objects:\n\n"
            for i, row in predictions.iterrows():
                name = row['name']
                confidence = row['confidence'] * 100
                result_text += f"{i+1}. {name} ({confidence:.1f}%)\n"
            
            self.results_label.config(text=result_text)
            self.status_var.set("Analysis complete")
        
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_var.set("Detection failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognitionApp(root)
    root.mainloop()
