from typing import Optional
import logging

import tkinter as tk
from PIL import ImageGrab, Image, ImageTk


log = logging.getLogger(__name__)


class RegionSelector:
    """
    Cross-platform region selector.
    Displays a dimmed overlay and highlights the selected region.
    Like a snippet tool, but the result is returned instead of being put to the clipboard.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.result: Optional[Image.Image] = None

        # These MUST be instance attributes to prevent garbage collection
        self.tk_img_full: Optional[ImageTk.PhotoImage] = None  # Full screenshot
        self.tk_img_overlay: Optional[ImageTk.PhotoImage] = None  # Dimming overlay
        self.tk_img_cropped: Optional[ImageTk.PhotoImage] = None  # "Clear" selection

    def _on_press(self, event, bbox: dict, canvas: tk.Canvas):
        """Handles the start of a mouse click/drag."""
        bbox["x0"], bbox["y0"] = event.x, event.y
        # Delete any previous selection drawings
        canvas.delete("selection")

    def _on_drag(self, event, bbox: dict, canvas: tk.Canvas, full_ss: Image.Image):
        """Handles the mouse drag, drawing the "clear" rectangle."""
        if bbox["x0"] is None or bbox["y0"] is None:
            return

        canvas.delete("selection")

        x0, y0 = bbox["x0"], bbox["y0"]
        x1, y1 = event.x, event.y

        # Ensure coordinates are ordered
        rx0, rx1 = sorted([x0, x1])
        ry0, ry1 = sorted([y0, y1])
        
        # Prevent zero-width/height crop
        if rx0 == rx1: rx1 += 1
        if ry0 == ry1: ry1 += 1

        try:
            # Crop the *original* screenshot
            cropped_img = full_ss.crop((rx0, ry0, rx1, ry1))

            # Convert to PhotoImage and store it
            # This reference MUST be kept to prevent garbage collection
            self.tk_img_cropped = ImageTk.PhotoImage(cropped_img)

            # Draw the "clear" (original) cropped image on top
            canvas.create_image(
                rx0, ry0, image=self.tk_img_cropped,
                anchor=tk.NW, tags="selection"
            )

            # Draw the selection border
            canvas.create_rectangle(
                rx0, ry0, rx1, ry1,
                outline="#00D1B2", width=2, tags="selection"
            )
        except Exception as e:
            log.debug("Failed to create/draw selection: %s", e)

    def _on_release(self, event, bbox: dict, root: tk.Tk, full_ss: Image.Image):
        """Handles mouse release, finalizing the selection."""
        if bbox["x0"] is None or bbox["y0"] is None:
            root.quit()
            return

        x0, y0 = bbox["x0"], bbox["y0"]
        x1, y1 = event.x, event.y
        
        rx0, rx1 = sorted([x0, x1])
        ry0, ry1 = sorted([y0, y1])

        try:
            # Crop the original screenshot to get the final result
            self.result = full_ss.crop((rx0, ry0, rx1, ry1))
        except Exception as e:
            log.debug("Image crop failed: %s", e)
        finally:
            root.quit()

    def select(self) -> Optional[Image.Image]:
        """Show region selector and return captured image."""
        
        # 1. Take the full screenshot *before* creating the window
        try:
            full_ss = ImageGrab.grab()
        except Exception as e:
            log.debug("Full ImageGrab failed: %s", e)
            return None

        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)

        # 2. Convert full screenshot to a Tkinter-compatible image
        # Store as instance variable to prevent garbage collection
        self.tk_img_full = ImageTk.PhotoImage(full_ss)

        # 3. Create the semi-transparent overlay image
        overlay_color = (0, 0, 0, 77)  # Black, 30% opaque
        overlay_img = Image.new("RGBA", full_ss.size, overlay_color)
        self.tk_img_overlay = ImageTk.PhotoImage(overlay_img)

        # 4. Create canvas and draw the images
        canvas = tk.Canvas(root, cursor="cross", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Draw the full screenshot (base layer)
        canvas.create_image(0, 0, image=self.tk_img_full, anchor=tk.NW)
        
        # Draw the dimming overlay on top
        canvas.create_image(0, 0, image=self.tk_img_overlay, anchor=tk.NW, tags="overlay")

        bbox = {"x0": None, "y0": None}

        # 5. Bind events, passing the full_ss to drag/release handlers
        canvas.bind(
            "<ButtonPress-1>", 
            lambda e: self._on_press(e, bbox, canvas)
        )
        canvas.bind(
            "<B1-Motion>", 
            lambda e: self._on_drag(e, bbox, canvas, full_ss)
        )
        canvas.bind(
            "<ButtonRelease-1>", 
            lambda e: self._on_release(e, bbox, root, full_ss)
        )
        root.bind("<Escape>", lambda e: root.quit())

        # Auto-cancel after timeout
        root.after(int(self.timeout * 1000), root.quit)

        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

        return self.result
