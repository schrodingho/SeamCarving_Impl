import numpy as np
import tkinter as tk
from tkinter import font as tkFont  # for convenience
from tkinter import messagebox
from PIL import Image, ImageTk
from pytorch_grad_cam.utils.image import show_cam_on_image
import dill

# TODO: close the window fix
# TODO: image size fix
class CamUI:
    def __init__(self, grayscale, image_float_array, cur_image):
        self.grayscale = grayscale
        self.image_float_array = image_float_array
        self.cur_image = cur_image
        self.width = image_float_array.shape[1]
        self.height = image_float_array.shape[0]
        self.modified_image = show_cam_on_image(self.image_float_array, self.grayscale, use_rgb=True)
        self.neighborhood_size = 25
        self.adjustment_scale = 0.3
        self.img_dst_path = "./cache/modified_image.png"
        self.mask_pkl_path = "./cache/modified_mask.pkl"

    def run_UI(self):
        def update_heatmap(x, y, increase):
            # Define the size of the neighborhood.
            self.neighborhood_size = int(neigh_scale.get())
            self.adjustment_scale = int(adj_scale.get()) / 100
            # Iterate over a neighborhood around the clicked point.
            for i in range(-self.neighborhood_size, self.neighborhood_size + 1):
                for j in range(-self.neighborhood_size, self.neighborhood_size + 1):
                    new_x = x + i
                    new_y = y + j
                    distance = np.sqrt(i ** 2 + j ** 2)
                    # Check if the new coordinates are within the image boundaries and within the circle.
                    if 0 <= new_x < self.width and 0 <= new_y < height and distance <= self.neighborhood_size:
                        # Calculate the adjustment factor based on distance from the center.
                        adjustment_factor = 1.0 - distance / self.neighborhood_size
                        # Adjust this scale for the desired effect.
                        adjustment_factor *= self.adjustment_scale

                        if increase:
                            self.grayscale[new_y, new_x] = min(self.grayscale[new_y, new_x] + adjustment_factor, 1.0)
                        else:
                            self.grayscale[new_y, new_x] = max(self.grayscale[new_y, new_x] - adjustment_factor, 0.0)

            # Generate new feature img
            modified_cam = self.grayscale
            self.modified_image = show_cam_on_image(self.image_float_array, modified_cam, use_rgb=True)
            modified_image_tk = ImageTk.PhotoImage(Image.fromarray(self.modified_image))

            label.config(image=modified_image_tk)
            label.image = modified_image_tk

        # Create a Tkinter window.
        root = tk.Tk()
        root.title("Feature Map Modification")
        height = self.image_float_array.shape[0]
        width = self.image_float_array.shape[1]
        root.geometry(f"{width + 30}x{height + 400}")

        # Create a label to display the image.
        label = tk.Label(root)
        label.pack()

        # Load the initial image.
        initial_image = ImageTk.PhotoImage(self.cur_image)
        label.config(image=initial_image)
        label.image = initial_image

        def save_image():
            if self.modified_image is not None:
                if self.img_dst_path:
                    dump_img = Image.fromarray(self.modified_image)
                    dump_img.save(self.img_dst_path)
                    dill.dump(self.grayscale, open(self.mask_pkl_path, 'wb'))
                    messagebox.showinfo("Success", "Saved successfully")
                    root.destroy()
                else:
                    messagebox.showerror("Error", "Please retry")

        elv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')

        # TIPS
        tips = tk.Label(root, height=5, width=width)
        tips['font'] = elv36
        tips['text'] = "Left click to increase the heatmap\nright click to decrease the heatmap"
        tips.pack()

        # Neighbourhood size scale bar
        neigh_scale = tk.Scale(root, from_=10, to=150, orient=tk.HORIZONTAL, label="Pen Size", length=200)
        neigh_scale['font'] = elv36
        neigh_scale.set(25)
        neigh_scale.pack()

        # Adjustment factor scale bar
        adj_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, label="Pen Strength", length=200)
        adj_scale['font'] = elv36
        adj_scale.set(30)
        adj_scale.pack()

        helv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')
        # Save button
        save_button = tk.Button(root, text="Save", command=save_image, height=10, width=20)
        save_button['font'] = helv36
        save_button.pack()

        # function to handle mouse clicks.
        # left click to increase, right click to decrease
        def on_canvas_left_click(event):
            x, y = event.x, event.y
            update_heatmap(x, y, increase=True)

        def on_canvas_right_click(event):
            x, y = event.x, event.y
            update_heatmap(x, y, increase=False)

        # Bind mouse click events to the label (image).
        label.focus_set()
        label.bind("<B1-Motion>", on_canvas_left_click)
        label.bind("<B3-Motion>", on_canvas_right_click)

        root.mainloop()

    def return_dst_path(self):
        return self.img_dst_path, self.mask_pkl_path




