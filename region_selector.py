from typing import Optional, Tuple
import logging

import tkinter as tk
from PIL import ImageGrab, Image, ImageTk


OVERLAY_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 77)
SELECTION_BORDER_COLOR: str = "#00D1B2"
SELECTION_BORDER_WIDTH: int = 2
SELECTION_TAG: str = "selection"
CANVAS_CURSOR: str = "cross"

log = logging.getLogger(__name__)


class RegionSelector:
    """
    Cross-platform region selector for capturing a screen area. 
    Displays a dimmed overlay; returns the selected area as a PIL Image.
    Like a snipping tool, but instead of pushing image to clipboard returns it.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout: float = timeout
        self.result: Optional[Image.Image] = None
        self._root: Optional[tk.Tk] = None
        self._coordinates = {"x0": None, "y0": None}

        # CRITICAL: These MUST be instance attributes to prevent Tkinter garbage collection.
        self._tk_img_full: Optional[ImageTk.PhotoImage] = None
        self._tk_img_overlay: Optional[ImageTk.PhotoImage] = None
        self._tk_img_cropped: Optional[ImageTk.PhotoImage] = None

    def _cleanup(self) -> None:
        """Destroys the tkinter root window and cleans up resources."""
        if self._root:
            try:
                self._root.destroy()
            except tk.TclError as e:
                log.debug(f"Error destroying Tkinter root: {e}")
        self._root = None
        self._tk_img_full = None
        self._tk_img_overlay = None
        self._tk_img_cropped = None

    @staticmethod
    def _get_normalized_bbox(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
        rx0, rx1 = sorted([x0, x1])
        ry0, ry1 = sorted([y0, y1])

        if rx0 == rx1:
            rx1 += 1
        if ry0 == ry1:
            ry1 += 1
            
        return rx0, ry0, rx1, ry1

    def _on_press(self, event: tk.Event, canvas: tk.Canvas) -> None:
        self._coordinates["x0"], self._coordinates["y0"] = event.x, event.y
        canvas.delete(SELECTION_TAG)

    def _draw_selection(self, canvas: tk.Canvas, full_ss: Image.Image, rx0: int, ry0: int, rx1: int, ry1: int) -> None:
        """Crops the original image and draws the clear selection and border onto the canvas"""
        try:
            cropped_img: Image.Image = full_ss.crop((rx0, ry0, rx1, ry1))
            
            # Store as instance variable to prevent garbage collection
            self._tk_img_cropped = ImageTk.PhotoImage(cropped_img)

            # Draw the clear (cropped) image
            canvas.create_image(rx0, ry0, image=self._tk_img_cropped, anchor=tk.NW, tags=SELECTION_TAG)
            
            # Draw the selection border
            canvas.create_rectangle(
                rx0, ry0, rx1, ry1, 
                outline=SELECTION_BORDER_COLOR, width=SELECTION_BORDER_WIDTH, tags=SELECTION_TAG
            )
        except Exception as e:
            log.debug(f"Failed to create/draw selection: {e}")

    def _on_drag(self, event: tk.Event, canvas: tk.Canvas, full_ss: Image.Image) -> None:
        x0, y0 = self._coordinates["x0"], self._coordinates["y0"]
        if x0 is None or y0 is None:
            return

        canvas.delete(SELECTION_TAG)

        x1, y1 = event.x, event.y
        rx0, ry0, rx1, ry1 = self._get_normalized_bbox(x0, y0, x1, y1)
        
        self._draw_selection(canvas, full_ss, rx0, ry0, rx1, ry1)

    def _on_release(self, event: tk.Event, full_ss: Image.Image) -> None:
        x0, y0 = self._coordinates["x0"], self._coordinates["y0"]
        if x0 is None or y0 is None:
            if self._root: self._root.quit()
            return

        x1, y1 = event.x, event.y
        rx0, ry0, rx1, ry1 = self._get_normalized_bbox(x0, y0, x1, y1)

        try:
            self.result = full_ss.crop((rx0, ry0, rx1, ry1))
        except Exception as e:
            log.error(f"Image crop failed on release: {e}")
        finally:
            if self._root: 
                self._root.quit()

    def _on_escape(self, event: tk.Event) -> None:
        """Handles the Escape key press to cancel the selection."""
        if self._root:
            log.debug("Escape key pressed. Cancelling selection.")
            self._root.quit()


    def select(self) -> Optional[Image.Image]:
        """Displays the region selector overlay and returns the captured image."""
        
        try:
            full_ss: Image.Image = ImageGrab.grab()
        except Exception as e:
            log.error(f"Full ImageGrab failed: {e}")
            return None

        self._root = tk.Tk()
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.01)

        self._tk_img_full = ImageTk.PhotoImage(full_ss)
        overlay_img = Image.new("RGBA", full_ss.size, OVERLAY_COLOR)
        self._tk_img_overlay = ImageTk.PhotoImage(overlay_img)

        canvas = tk.Canvas(self._root, cursor=CANVAS_CURSOR, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.create_image(0, 0, image=self._tk_img_full, anchor=tk.NW)
        canvas.create_image(0, 0, image=self._tk_img_overlay, anchor=tk.NW, tags="overlay")

        self._root.attributes("-alpha", 1.0) 

        canvas.bind("<ButtonPress-1>", lambda e: self._on_press(e, canvas))
        canvas.bind("<B1-Motion>", lambda e: self._on_drag(e, canvas, full_ss))
        canvas.bind("<ButtonRelease-1>", lambda e: self._on_release(e, full_ss))
        
        self._root.bind_all("<Escape>", self._on_escape)
        self._root.after(int(self.timeout * 1000), self._root.quit)

        try:
            self._root.mainloop()
        except Exception as e:
            log.error(f"Error during Tkinter mainloop: {e}")
        finally:
            self._cleanup()

        return self.result
