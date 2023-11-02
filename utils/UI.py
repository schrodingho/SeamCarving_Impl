import numpy as np
import tkinter as tk
from tkinter import font as tkFont  # for convenience
from tkinter import messagebox
from PIL import Image, ImageTk
from pytorch_grad_cam.utils.image import show_cam_on_image
import dill

class CamUI:
    def __init__(self, grayscale, image_float_array, cur_image):
        self.grayscale = grayscale
        self.image_float_array = image_float_array
        self.cur_image = cur_image
        self.width = image_float_array.shape[1]
        self.height = image_float_array.shape[0]
        self.modified_image = show_cam_on_image(self.image_float_array, self.grayscale, use_rgb=True)
        # if not exist cache folder, create one, use os
        self.img_dst_path = "./cache/modified_image.png"
        self.mask_pkl_path = "./cache/modified_mask.pkl"


    def run_UI(self):
        def update_heatmap(x, y, increase):
            # Define the size of the neighborhood (you can adjust this as needed).
            neighborhood_size = 25

            # Iterate over a neighborhood around the clicked point.
            for i in range(-neighborhood_size, neighborhood_size + 1):
                for j in range(-neighborhood_size, neighborhood_size + 1):
                    new_x = x + i
                    new_y = y + j
                    distance = np.sqrt(i ** 2 + j ** 2)
                    # Check if the new coordinates are within the image boundaries and within the circle.
                    if 0 <= new_x < self.width and 0 <= new_y < height and distance <= neighborhood_size:
                        # Calculate the adjustment factor based on distance from the center.
                        adjustment_factor = 1.0 - distance / neighborhood_size
                        adjustment_factor *= 0.5  # Adjust this factor for the desired effect.

                        if increase:
                            self.grayscale[new_y, new_x] = min(self.grayscale[new_y, new_x] + adjustment_factor, 1.0)
                        else:
                            self.grayscale[new_y, new_x] = max(self.grayscale[new_y, new_x] - adjustment_factor, 0.0)

            modified_cam = self.grayscale
            self.modified_image = show_cam_on_image(self.image_float_array, modified_cam, use_rgb=True)
            modified_image_tk = ImageTk.PhotoImage(Image.fromarray(self.modified_image))

            label.config(image=modified_image_tk)
            label.image = modified_image_tk

        # Create a Tkinter window.
        root = tk.Tk()
        root.title("GradCAM Heatmap Modification")
        height = self.image_float_array.shape[0]
        width = self.image_float_array.shape[1]
        root.geometry(f"{width + 10}x{height + 100}")

        # Create a label to display the image.
        label = tk.Label(root)
        label.pack()

        # Load the initial image.
        initial_image = ImageTk.PhotoImage(self.cur_image)
        label.config(image=initial_image)
        label.image = initial_image

        def save_image():
            # global self.modified_image
            if self.modified_image is not None:
                # self.img_dst_path = tk.filedialog.asksaveasfilename(defaultextension=".png")
                if self.img_dst_path:
                    dump_img = Image.fromarray(self.modified_image)
                    dump_img.save(self.img_dst_path)
                    dill.dump(self.grayscale, open(self.mask_pkl_path, 'wb'))
                    messagebox.showinfo("Success", "Saved successfully")
                    root.destroy()
                else:
                    messagebox.showerror("Error", "Please retry")
                    # raise ValueError("Please specify the path to save the image")

        helv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')
        save_button = tk.Button(root, text="Save", command=save_image, height=10, width=20)
        save_button['font'] = helv36
        save_button.pack()

        # # Create a function to handle mouse clicks.
        def on_canvas_click(event):
            x, y = event.x, event.y
            update_heatmap(x, y, increase=True)

        def on_canvas_right_click(event):
            x, y = event.x, event.y
            update_heatmap(x, y, increase=False)

        # Bind mouse click events to the label (image).
        label.focus_set()
        label.bind("<B1-Motion>", on_canvas_click)
        label.bind("<B3-Motion>", on_canvas_right_click)

        # Initialize the grayscale map.
        # Run the Tkinter main loop.
        root.mainloop()

    def return_dst_path(self):
        return self.img_dst_path, self.mask_pkl_path




