import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.utils as image
from keras.applications.mobilenet_v2 import preprocess_input
import argparse
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Tkinter integration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GPU Memory Management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs detected. Running on CPU.")

# Class mappings from the original model
CLASS_LABELS = [
    '2_blade_rotor_1', '2_blade_rotor_2', '3_long_blades_rotor', 
    '3_short_blade_rotor_1', '3_short_blade_rotor_2', 'Bird', 
    'Bird+2_Blade_rotor_1', 'Bird+2_Blade_rotor_2', 'drone_1', 'drone_2'
]

# Group mapping for binary classification
GROUP_MAPPING = {
    '2_blade_rotor_1': 'drone', '2_blade_rotor_2': 'drone',
    '3_long_blades_rotor': 'drone', '3_short_blade_rotor_1': 'drone',
    '3_short_blade_rotor_2': 'drone', 'Bird': 'bird',
    'Bird+2_Blade_rotor_1': 'ambiguous', 'Bird+2_Blade_rotor_2': 'ambiguous',
    'drone_1': 'drone', 'drone_2': 'drone'
}

# Group confidence weighting
GROUP_CONFIDENCE = {
    '2_blade_rotor_1': 1.0, '2_blade_rotor_2': 1.0,
    '3_long_blades_rotor': 1.0, '3_short_blade_rotor_1': 1.0,
    '3_short_blade_rotor_2': 1.0, 'Bird': 1.0,
    'Bird+2_Blade_rotor_1': {'bird': 0.6, 'drone': 0.4},
    'Bird+2_Blade_rotor_2': {'bird': 0.6, 'drone': 0.4},
    'drone_1': 1.0, 'drone_2': 1.0
}

def focal_loss_with_smoothing(gamma=2.0, alpha=0.25, label_smoothing=0.1, num_classes=10):
    """Custom focal loss function for model loading."""
    def loss_fn(y_true, y_pred):
        y_true = y_true * (1 - label_smoothing) + label_smoothing / num_classes
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1. - y_pred, gamma) * y_true
        focal_loss = weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return loss_fn

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def aggregate_predictions(pred_probs, class_labels, group_mapping, group_confidence):
    """Aggregate probabilities for binary classification."""
    group_scores = {'bird': 0.0, 'drone': 0.0}
    for i, label in enumerate(class_labels):
        prob = pred_probs[i]
        group = group_mapping.get(label, 'unknown')
        if group in ['bird', 'drone']:
            group_scores[group] += prob * 1.0
        elif group == 'ambiguous':
            conf = group_confidence.get(label, {'bird': 0.5, 'drone': 0.5})
            group_scores['bird'] += prob * conf['bird']
            group_scores['drone'] += prob * conf['drone']
    return group_scores['bird'], group_scores['drone']

def visualize_prediction(img, predictions, top_n=3):
    """Create a visualization of the image and prediction results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    indices = np.argsort(predictions)[-top_n:][::-1]
    labels = [CLASS_LABELS[i] for i in indices]
    probs = [predictions[i] * 100 for i in indices]
    bars = ax2.barh(labels, probs, color='skyblue')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title(f'Top {top_n} Predictions')
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{probs[i]:.1f}%', va='center')
    bird_score, drone_score = aggregate_predictions(predictions, CLASS_LABELS, GROUP_MAPPING, GROUP_CONFIDENCE)
    binary_pred = 'Bird' if bird_score > drone_score else 'Drone'
    binary_conf = max(bird_score, drone_score) / (bird_score + drone_score) * 100
    plt.figtext(0.5, 0.01, f'Overall: {binary_pred} ({binary_conf:.1f}% confidence)', 
                ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.2))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

def predict_image(model_path, image_path, output_dir=None, top_n=3, show_plot=True, keep_figure=False):
    """Predict the class of a single image."""
    custom_objects = {'loss_fn': focal_loss_with_smoothing()}
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    try:
        img_array, img = preprocess_image(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    print("Making prediction...")
    predictions = model.predict(img_array)[0]
    temperature = 1.5
    predictions = np.exp(np.log(predictions) / temperature)
    predictions = predictions / np.sum(predictions)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_LABELS[predicted_class_idx]
    bird_score, drone_score = aggregate_predictions(predictions, CLASS_LABELS, GROUP_MAPPING, GROUP_CONFIDENCE)
    binary_prediction = 'Bird' if bird_score > drone_score else 'Drone'
    fig = visualize_prediction(img, predictions, top_n)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        filename, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{filename}_prediction.png")
        fig.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        text_output = os.path.join(output_dir, f"{filename}_results.txt")
        with open(text_output, 'w') as f:
            f.write(f"Detailed Class: {predicted_class}\n")
            f.write(f"Binary Class: {binary_prediction}\n")
        print(f"Saved detailed results to {text_output}")
    if show_plot:
        plt.show()
    if keep_figure:
        return predicted_class, binary_prediction, predictions, fig
    else:
        plt.close(fig)
        return predicted_class, binary_prediction, predictions, None

def batch_predict(model_path, image_dir, output_dir=None, progress_callback=None):
    """Predict classes for multiple images in a directory."""
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} is not a directory")
        return None
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    if not image_files:
        print(f"No image files found in {image_dir}")
        return None
    print(f"Found {len(image_files)} images. Starting batch prediction...")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, img_path in enumerate(image_files):
        if progress_callback:
            progress_callback(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        result = predict_image(model_path, img_path, output_dir, show_plot=False, keep_figure=False)
        if result:
            predicted_class, binary_prediction, _, _ = result
            results.append({'file': img_path, 'detailed_class': predicted_class, 'binary_class': binary_prediction})
            if progress_callback:
                progress_callback(f"{os.path.basename(img_path)}: {binary_prediction} ({predicted_class})")
    if output_dir and results:
        summary_path = os.path.join(output_dir, "batch_prediction_summary.csv")
        with open(summary_path, 'w') as f:
            f.write("file,detailed_class,binary_class\n")
            for r in results:
                f.write(f"{r['file']},{r['detailed_class']},{r['binary_class']}\n")
        print(f"Saved batch prediction summary to {summary_path}")
    if progress_callback:
        progress_callback("Batch prediction completed.")
    return results

class ClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bird/Drone Image Classifier")
        self.geometry("800x600")
        self.queue = queue.Queue()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.single_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text="Single Prediction")
        self.notebook.add(self.batch_tab, text="Batch Prediction")
        
        self.setup_single_tab()
        self.setup_batch_tab()
        
        # Log area
        self.log_text = tk.Text(self, height=10)
        self.log_text.pack(fill='both', expand=True)
        self.log_text.config(state='disabled')
        
        # Start checking the queue
        self.check_queue()

    def setup_single_tab(self):
        """Set up the single prediction tab."""
        tk.Label(self.single_tab, text="Model File:").grid(row=0, column=0, padx=5, pady=5)
        self.single_model_var = tk.StringVar()
        tk.Entry(self.single_tab, textvariable=self.single_model_var, width=50).grid(row=0, column=1)
        tk.Button(self.single_tab, text="Browse", command=self.browse_single_model).grid(row=0, column=2)
        
        tk.Label(self.single_tab, text="Image File:").grid(row=1, column=0, padx=5, pady=5)
        self.single_image_var = tk.StringVar()
        tk.Entry(self.single_tab, textvariable=self.single_image_var, width=50).grid(row=1, column=1)
        tk.Button(self.single_tab, text="Browse", command=self.browse_single_image).grid(row=1, column=2)
        
        tk.Label(self.single_tab, text="Output Directory (optional):").grid(row=2, column=0, padx=5, pady=5)
        self.single_output_var = tk.StringVar()
        tk.Entry(self.single_tab, textvariable=self.single_output_var, width=50).grid(row=2, column=1)
        tk.Button(self.single_tab, text="Browse", command=self.browse_single_output).grid(row=2, column=2)
        
        tk.Label(self.single_tab, text="Top N Predictions:").grid(row=3, column=0, padx=5, pady=5)
        self.top_n_var = tk.IntVar(value=3)
        tk.Spinbox(self.single_tab, from_=1, to=10, textvariable=self.top_n_var).grid(row=3, column=1)
        
        self.predict_button = tk.Button(self.single_tab, text="Predict", command=self.start_single_prediction)
        self.predict_button.grid(row=4, column=1, pady=10)
        
        self.plot_frame = tk.Frame(self.single_tab)
        self.plot_frame.grid(row=5, column=0, columnspan=3, sticky='nsew')

    def setup_batch_tab(self):
        """Set up the batch prediction tab."""
        tk.Label(self.batch_tab, text="Model File:").grid(row=0, column=0, padx=5, pady=5)
        self.batch_model_var = tk.StringVar()
        tk.Entry(self.batch_tab, textvariable=self.batch_model_var, width=50).grid(row=0, column=1)
        tk.Button(self.batch_tab, text="Browse", command=self.browse_batch_model).grid(row=0, column=2)
        
        tk.Label(self.batch_tab, text="Image Directory:").grid(row=1, column=0, padx=5, pady=5)
        self.batch_image_dir_var = tk.StringVar()
        tk.Entry(self.batch_tab, textvariable=self.batch_image_dir_var, width=50).grid(row=1, column=1)
        tk.Button(self.batch_tab, text="Browse", command=self.browse_batch_image_dir).grid(row=1, column=2)
        
        tk.Label(self.batch_tab, text="Output Directory (optional):").grid(row=2, column=0, padx=5, pady=5)
        self.batch_output_var = tk.StringVar()
        tk.Entry(self.batch_tab, textvariable=self.batch_output_var, width=50).grid(row=2, column=1)
        tk.Button(self.batch_tab, text="Browse", command=self.browse_batch_output).grid(row=2, column=2)
        
        self.batch_button = tk.Button(self.batch_tab, text="Start Batch Prediction", command=self.start_batch_prediction)
        self.batch_button.grid(row=3, column=1, pady=10)

    def browse_single_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.single_model_var.set(path)

    def browse_single_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if path:
            self.single_image_var.set(path)

    def browse_single_output(self):
        path = filedialog.askdirectory()
        if path:
            self.single_output_var.set(path)

    def browse_batch_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.batch_model_var.set(path)

    def browse_batch_image_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_image_dir_var.set(path)

    def browse_batch_output(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_output_var.set(path)

    def start_single_prediction(self):
        """Start a single image prediction in a background thread."""
        model_path = self.single_model_var.get()
        image_path = self.single_image_var.get()
        output_dir = self.single_output_var.get() or None
        top_n = self.top_n_var.get()
        if not model_path or not image_path:
            self.log("Please select model and image files.")
            return
        self.predict_button.config(state='disabled')
        self.log("Starting single prediction...")
        thread = threading.Thread(target=self.run_single_prediction, args=(model_path, image_path, output_dir, top_n))
        thread.start()

    def run_single_prediction(self, model_path, image_path, output_dir, top_n):
        """Run single prediction and send results to queue."""
        try:
            result = predict_image(model_path, image_path, output_dir, top_n, show_plot=False, keep_figure=True)
            if result:
                predicted_class, binary_prediction, predictions, fig = result
                self.queue.put(('single_result', fig, predicted_class, binary_prediction, predictions))
            else:
                self.queue.put(('error', "Prediction failed."))
        except Exception as e:
            self.queue.put(('error', str(e)))

    def start_batch_prediction(self):
        """Start a batch prediction in a background thread."""
        model_path = self.batch_model_var.get()
        image_dir = self.batch_image_dir_var.get()
        output_dir = self.batch_output_var.get() or None
        if not model_path or not image_dir:
            self.log("Please select model file and image directory.")
            return
        self.batch_button.config(state='disabled')
        self.log("Starting batch prediction...")
        thread = threading.Thread(target=self.run_batch_prediction, args=(model_path, image_dir, output_dir))
        thread.start()

    def run_batch_prediction(self, model_path, image_dir, output_dir):
        """Run batch prediction and send progress to queue."""
        try:
            batch_predict(model_path, image_dir, output_dir, 
                          progress_callback=lambda msg: self.queue.put(('progress', msg)))
        except Exception as e:
            self.queue.put(('error', str(e)))

    def check_queue(self):
        """Check the queue for updates from background threads."""
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == 'single_result':
                    _, fig, predicted_class, binary_prediction, predictions = item
                    self.display_single_result(fig, predicted_class, binary_prediction, predictions)
                elif item[0] == 'progress':
                    self.log(item[1])
                elif item[0] == 'error':
                    self.log(f"Error: {item[1]}")
        except queue.Empty:
            pass
        self.after(100, self.check_queue)

    def display_single_result(self, fig, predicted_class, binary_prediction, predictions):
        """Display single prediction results in the GUI."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.log(f"Detailed Class: {predicted_class}")
        self.log(f"Binary Class: {binary_prediction}")
        self.predict_button.config(state='normal')

    def log(self, message):
        """Append a message to the log area."""
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.config(state='disabled')
        self.log_text.see('end')

if __name__ == "__main__":
    # Remove argparse for GUI version; keep it commented for reference
    
    parser = argparse.ArgumentParser(description="Bird/Drone Image Classifier")
    parser.add_argument("--model", type=str, default="model_outputs/best_model.h5", 
                        help="Path to the trained model file")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    single_parser = subparsers.add_parser("predict", help="Predict a single image")
    single_parser.add_argument("image", type=str, help="Path to the image file")
    single_parser.add_argument("--output", type=str, help="Directory to save outputs")
    single_parser.add_argument("--top", type=int, default=3, help="Number of top predictions to show")
    batch_parser = subparsers.add_parser("batch", help="Predict a batch of images")
    batch_parser.add_argument("dir", type=str, help="Directory containing images")
    batch_parser.add_argument("--output", type=str, help="Directory to save outputs")
    args = parser.parse_args()
    if args.command == "predict":
        predict_image(args.model, args.image, args.output, args.top)
    elif args.command == "batch":
        batch_predict(args.model, args.dir, args.output)
    else:
        parser.print_help()
    
    app = ClassifierGUI()
    app.mainloop()