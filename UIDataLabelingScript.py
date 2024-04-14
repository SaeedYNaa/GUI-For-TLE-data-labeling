# -*- coding: utf-8 -*-
# Advanced zoom for images of various types from small to huge up to several GB
import math
import os
import tkinter as tk
import cv2 as cv
from tkinter import ttk
from PIL import Image, ImageTk
from UTILS.utils import DoublyLinkedPoints
import glob
from tkinter import messagebox
import uuid
import numpy as np
from UTILS.utils import calculate_distance
import json



class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """
    def __init__(self, placeholder, path=None, output_path=None):
        """ Initialize the ImageFrame """
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.LANCZOS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.folder_path = path
        self.output_path = output_path

        self.master = placeholder
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(self.master)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas_width = 800
        self.canvas_height = 800
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set,
                                width=self.canvas_width,
                                height=self.canvas_height
                                )
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)

        # control point tracker
        self.control_point_tracker = DoublyLinkedPoints()
        self.control_point_tracker_system = {}  # contains tracker for each image for each bowl
        self.changed_control_points = False
        self.current_working_control_point = None
        self.thickness = 2
        self.drawing = True  # true if mouse is pressed

        # Bowls
        self.bowl_dirs = []
        self.bowl_idx = -1
        self.image_idx = -1
        self.bowl_images = []
        self.saved_image_path = ""
        self.metadata_file = "metadata.json"
        self.dir_img_state = {}
        self.img_path = None
        self.img = None
        self.img_id = None
        self.scale = 1.0
        self.zoom_in = False

        self.cwd = None
        self.img_name = None
        self.total_wheeling = 0
        self.metadata = {}
        self.polygon = []  # vertices of the current (drawing, red) polygon
        self.edge = None  # current edge of the new polygon
        self.dash = (1, 1)  # dash pattern
        self.radius_stick = 10  # distance where line sticks to the polygon's staring point
        # mapping of points (x, y) to tags (objects)
        self.point_to_tag = {}
        self.line_to_tag = {}

        # load bowls:
        self.load_bowls()
        self.load_image()

        self.color_hole = {'draw': 'white',  # draw hole color
                           'point': '#8B0000',  # point hole color
                           'back': 'cyan',  # background hole color
                           'stipple': 'gray50'}  # stipple value for hole

        self.radius_circle = 5  # radius of the sticking circle
        self.tag_curr_edge_start = '1st_edge'  # starting edge of the current polygon
        self.tag_curr_edge = 'edge'  # edges of the polygon
        self.tag_curr_edge_id = 'edge_id'  # part of unique ID of the current edge
        self.tag_const = 'poly'  # constant tag for polygon
        self.tag_poly_line = 'poly_line'  # edge of the polygon
        self.tag_curr_circle = 'circle'  # sticking circle tag for the current polyline

        self.master.bind("<KeyPress>", self.on_key_press)

        # menu bar
        self.toolbar = tk.Menu(placeholder)
        self.toolbar.add_command(label='Prev', command=self.prev_image)
        self.toolbar.add_command(label='Next', command=self.next_image)
        self.toolbar.add_command(label='Prev Bowl', command=self.prev_bowl_dir)
        self.toolbar.add_command(label='Next Bowl', command=self.next_bowl_dir)
        self.toolbar.add_command(label='Save', command=self.save_image)

        placeholder.config(menu=self.toolbar)
        # Bind events to the Canvas

        print(f'binding: ', end='')
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        # self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        # self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        # ZOOMING AND PAN
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        # DRAWING
        self.canvas.bind('<ButtonPress-1>', self.set_edge)  # set new edge
        self.canvas.bind("<Button-3>", self.__move_from)
        self.canvas.bind("<B3-Motion>", self.__move_to)
        # self.canvas.bind("<B1-Motion>", self.on_mouse_up)
        self.canvas.bind("<Control-z>", self.right_click)

        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image


        self.__min_side = min(self.canvas_width, self.canvas_height)  # get the smaller image side
        # Create image pyramid
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.canvas_width, self.canvas_height) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.canvas_width, self.canvas_height), width=0)
        print(f'CALLing ', end='')
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def set_dash(self, x, y):
        """ Set dash for edge segment """
        self.canvas.itemconfigure(self.edge, dash='')  # set solid line

    def on_key_press(self, event):
        """
        when the :keyword "n" is pressed, new polygon will be created.
        :param event:
        :return:
        """
        if event.char.lower() == "n":
            # append the current polygon and create new one:
            self.control_point_tracker_system[self.bowl_idx][self.image_idx][0].append(self.control_point_tracker.copy())

            self.control_point_tracker = DoublyLinkedPoints()
            print(f'CALLING - on key pressed ', end='')
            self.__show_image()

    def load_bowls(self):
        self.bowl_dirs = sorted(
            [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))])

    def set_thickness(self, val):
        self.thickness = int(val)

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        print(f'CALLING - scrollx')
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        print(f'CALLING - scrolly')

        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        # Draw all polygons of the current image


        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly

        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]

        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]

        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                (int(x1 / self.__scale), int(y1 / self.__scale),
                                 int(x2 / self.__scale), int(y2 / self.__scale)))

            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)

            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        if event.state & 0x4 and self.zoom_in:
            self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        if event.state & 0x4 and self.zoom_in:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            print(f'CALLING - move to')

            self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        if event.state & 0x4:
            x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
            y = self.canvas.canvasy(event.y)
            if self.outside(x, y): return  # zoom only inside image area
            scale = 1.0
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down, smaller
                # if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                if self.imscale <= 1.2:
                    self.zoom_in = False
                    return
                self.imscale /= self.__delta
                scale        /= self.__delta
                self.zoom_in = True
            if event.num == 4 or event.delta == 120:  # scroll up, bigger
                # i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
                # if i < self.imscale: return  # 1 pixel is bigger than the visible area
                if self.imscale > 3:
                    return
                self.imscale *= self.__delta
                scale        *= self.__delta
                self.zoom_in = True
            # Take appropriate image from the pyramid
            k = self.imscale * self.__ratio  # temporary coefficient
            self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
            self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
            #
            self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
            # Redraw some figures before showing image on the screen
            print(f'CALLING - whjeel', end='')

            self.__show_image()

    # Functionality
    def set_edge(self, event):
        # self.motion(event)  # generate motion event. It's needed for menu bar, bug otherwise!

        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)

        bbox = self.canvas.coords(self.container)  # get image area
        x1 = round((x - bbox[0]) / self.imscale)  # get real (x,y) on the image without zoom
        y1 = round((y - bbox[1]) / self.imscale)
        changed, self.current_working_control_point = self.clicked_point_in_radius(x1, y1, radius=5 // self.imscale,
                                                                                   update_last=False)
        if changed:
            self.drawing = False
            self.changed_control_points = False
            print(f'changing point: {x1, y1}')

        else:
            color = self.color_hole  # set color palette
            # Draw sticking circle
            tag = f'{self.tag_curr_edge}-{uuid.uuid4().hex}'
            self.draw_edge(color, x, y, tag)  # continue drawing polygon, set new edge

    def draw_polygon(self, polygon, color):
        """ Draw polygon on the canvas """
        if polygon is not None:
            # Calculate coordinates of vertices on the zoomed image
            bbox = self.canvas.coords(self.container)  # get image area
            vertices = list(map((lambda i: (i[0] * self.imscale + bbox[0],
                                            i[1] * self.imscale + bbox[1])), polygon))
            # Create identification tag
            # Create polyline. 2nd tag is ALWAYS a unique tag ID.
            # self.draw_ovals()
            for j in range(0, len(vertices) - 1):
                uuidd = uuid.uuid4().hex
                tag = f'{self.tag_poly_line}-{uuidd}'
                id = f'{tuple(map(int, polygon[j]))}-{tuple(map(int, polygon[j + 1]))}'
                if id not in self.line_to_tag:
                    self.line_to_tag[id] = tag
                self.canvas.create_line(vertices[j], vertices[j + 1], width=self.thickness,
                                        fill=color['draw'], tags=self.line_to_tag[id])

                print(f'INSERING: line {id} - {self.line_to_tag[id]}')


    def update_img(self):
        polys = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0].copy()
        polys.append(self.control_point_tracker.copy())
        print(f'TOTAL POLYS top draw: {len(polys)}')
        for poly in polys:
            self.draw_polygon(poly.get_points_as_array(), color=self.color_hole)
        # draw current poly:
        # draw all points
        # self.draw_ovals()
        print(f'Current tags: \n line: {self.line_to_tag}\n point: {self.point_to_tag}')


    def draw_ovals(self):
        all_points = self.control_point_tracker_system[self.bowl_idx][self.image_idx][2]
        # all_points.append(self.point_to_tag.copy())
        for points in all_points:
            for point, tag in points.items():
                x, y = point

                self.canvas.create_oval(x - self.radius_circle, y - self.radius_circle,
                                        x + self.radius_circle, y + self.radius_circle,
                                        tags=tag,
                                        fill='red',
                                        width=self.thickness)

    def draw_edge(self, color, x, y, tags=None):
        self.canvas.create_oval(x - self.radius_circle*self.imscale, y - self.radius_circle*self.imscale,
                                x + self.radius_circle*self.imscale, y + self.radius_circle*self.imscale,
                                fill='red',
                                tags=tags,
                                width=self.thickness)

        bbox = self.canvas.coords(self.container)  # get image area
        x1 = round((x - bbox[0]) / self.imscale)  # get real (x,y) on the image without zoom
        y1 = round((y - bbox[1]) / self.imscale)
        self.control_point_tracker.insert((x1, y1))
        if (x1, y1) not in self.point_to_tag:
            print(f'INSERTING: point: {(x1, y1)} - {tags}')
            self.point_to_tag[(x1, y1)] = tags

        print(f'CALLING - DRAW EDGE', end='')
        self.update_img()
        self.__show_image()

    def clicked_point_in_radius(self, x, y, radius=5, update_last=False):
        """
        check if point (x, y) is in radius 3p from some control point
        :param x:
        :param y:
        :return:
        """
        tmp = self.control_point_tracker.head
        index = 0
        while tmp is not None:
            center_x, center_y = tmp.data
            if (x >= center_x - radius and x <= center_x + radius) and \
                    (y >= center_y - radius and y <= center_y + radius):

                if update_last:
                    self.control_point_tracker.last = tmp

                return True, index
            tmp = tmp.next
            index += 1

        return False, -1

    def right_click(self, event):
        # first approach:
        # when pressing right click, delete last point.
        if self.control_point_tracker.len() > 0:
            popped = self.control_point_tracker.pop()
            if popped.prev: # delete also the line associate with this point
                prev_popped = popped.prev
                id = f'{prev_popped.data}-{popped.data}'
                self.canvas.delete(self.line_to_tag[id])
                print(f'DELETING: line: {id}- {self.line_to_tag[id]}')
                self.line_to_tag.pop(id)

            self.canvas.delete(self.point_to_tag[popped.data])
            self.point_to_tag.pop(popped.data)
            # self.update_img()
        else:
            if len(self.control_point_tracker_system[self.bowl_idx][self.image_idx][0]) > 1:
                print(f'DELETING: pop linked: {len(self.control_point_tracker_system[self.bowl_idx][self.image_idx][0])}')
                self.control_point_tracker = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0].pop()
                # self.update_img()
        print(f'calling - right_click', end='')
        self.update_img()
        self.__show_image()

    def load_env_vars(self):
        """
        this function is responsible for init the prop. variables, images paths, and the point tracker
        :return curr_working_dir: the current_working_dir path
        """
        if self.bowl_idx == -1 or self.image_idx < 0:
            self.bowl_idx = 0
            self.image_idx = 0
            # check if first initialization:
            if self.bowl_idx not in self.control_point_tracker_system.keys():
                self.control_point_tracker_system[self.bowl_idx] = {self.image_idx: [[DoublyLinkedPoints()], [{}], [{}]]}

        if len(self.bowl_dirs) == 0:
            tk.messagebox.showwarning("Error", "No bowls found in the specified folder.")
            return

        if len(self.bowl_images) == 0:
            self.bowl_images = glob.glob(os.path.join(self.folder_path, self.bowl_dirs[self.bowl_idx], '*'))
            if len(self.bowl_images) == 0:
                tk.messagebox.showwarning("Error", "No images found for the specified bowl.")
                return

        self.cwd, self.img_name = self.get_dir_img_names(self.bowl_idx, self.image_idx)

        if self.cwd not in self.metadata:
            self.metadata[self.cwd] = {}

        if self.bowl_idx not in self.control_point_tracker_system.keys():
            self.control_point_tracker_system[self.bowl_idx] = {}

        if self.image_idx not in self.control_point_tracker_system[self.bowl_idx]:
            self.control_point_tracker_system[self.bowl_idx][self.image_idx] = [[DoublyLinkedPoints()], [{}], [{}]]

        # retrieve last drawing lines, and corresponiding tags.
        self.control_point_tracker = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0][-1]
        self.line_to_tag = self.control_point_tracker_system[self.bowl_idx][self.image_idx][1][-1]
        self.point_to_tag = self.control_point_tracker_system[self.bowl_idx][self.image_idx][2][-1]

        if self.cwd not in self.dir_img_state:
            self.dir_img_state[self.cwd] = {}

        return self.cwd

    def load_image(self):
        # load the environment variable
        curr_working_dir = self.load_env_vars()
        img_path = self.bowl_images[self.image_idx]

        self.master.title(img_path)
        self.saved_image_path = os.path.join(self.output_path, f"{self.bowl_dirs[self.bowl_idx]}_annotated")
        if not os.path.exists(self.saved_image_path):
            os.makedirs(self.saved_image_path)

        # check if the image is a new image on the editor
        if self.img_name not in self.dir_img_state[curr_working_dir]:
            self.img_path = img_path
            self.img = cv.imread(img_path)
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            self.img = cv.resize(self.img, dsize=(self.canvas_width, self.canvas_height))
            self.img = Image.fromarray(self.img)
            self.dir_img_state[curr_working_dir][self.img_name] = {"current_img": self.img.copy(),
                                                                   "copy": self.img.copy(),
                                                                   "original_img": self.img.copy()}

        else:
            self.img = self.dir_img_state[curr_working_dir][self.img_name]["current_img"].copy()

        self.__pyramid = [self.img]

    def next_image(self):
        self.reset_canvas()

        self.image_idx += 1
        if self.image_idx >= len(self.bowl_images):
            tk.messagebox.showwarning("Error", "No more images found for the specified bowl.")
            self.image_idx = len(self.bowl_images) - 1
            self.control_point_tracker = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0][-1]
            return


        self.load_image()
        print(f'NEXT IMAGE: ', end='')
        self.__show_image()

    def prev_image(self):
        self.reset_canvas()

        self.image_idx -= 1
        if self.image_idx < 0:
            tk.messagebox.showwarning("Error", "No previous images found for the specified bowl.")
            self.image_idx = 0
            self.control_point_tracker = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0][-1]
            return

        self.load_image()
        print(f'prev IMAGE: ', end='')

        self.__show_image()

    def next_bowl_dir(self):
        self.reset_canvas()

        # creating control point state for this bowl:

        self.bowl_idx += 1
        self.image_idx = 0
        self.bowl_images = []
        self.changed_control_points = False
        if self.bowl_idx >= len(self.bowl_dirs):
            tk.messagebox.showwarning("Error", "Last Bowl Directory")
            self.bowl_idx -= 1

        self.load_image()
        self.__show_image()

    def prev_bowl_dir(self):
        self.reset_canvas()

        if self.bowl_idx > 0:
            self.bowl_idx -= 1
            self.image_idx = 0
            self.bowl_images = []
            self.changed_control_points = False
        else:
            tk.messagebox.showwarning("Error", "First Bowl Folder!")
            self.bowl_idx = 0

        self.control_point_tracker = DoublyLinkedPoints()
        self.reset_canvas()
        self.load_image()
        self.__show_image()

    def save_image(self):
        if self.img is None:
            return

        img_path = self.bowl_images[self.image_idx]
        img_name = os.path.basename(img_path)
        cwd = self.bowl_dirs[self.bowl_idx]
        if img_name not in self.metadata[cwd]:
            self.metadata[cwd][img_name] = []

        # create a mask image
        mask = np.zeros((self.img.size[0], self.img.size[1], 3), dtype='uint8')
        other_polys = self.control_point_tracker_system[self.bowl_idx][self.image_idx][0]
        for poly in other_polys:
            all_points = poly.get_points_as_array()
            cv.polylines(mask, np.int32([all_points]), False, (255, 255, 255), thickness=self.thickness)
            self.metadata[cwd][img_name].append(all_points.tolist())

        curr_working_points = self.control_point_tracker.get_points_as_array()
        cv.polylines(mask, np.int32([curr_working_points]), False, (255, 255, 255), thickness=self.thickness)
        self.metadata[cwd][img_name].append(curr_working_points.tolist())

        cv.imshow("mask", mask)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # add the distance
        distance_transform_img = calculate_distance(mask)
        cv.normalize(distance_transform_img, dst=distance_transform_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                     dtype=cv.CV_8U)

        cv.imshow("distance", distance_transform_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # save the image
        print(f"SAVING - results: {os.path.join(self.saved_image_path, img_name)}")
        cv.imwrite(os.path.join(self.saved_image_path, img_name), mask)
        # print(f'dist image: {distance_transform_img.shape}')
        # cv.imwrite(os.path.join(self.saved_image_path, f'dst_{img_name}'), distance_transform_img)
        # add the metadata format
        with open(self.metadata_file, mode='w') as f:
            json.dump(self.metadata, f, indent=None, separators=(', ', ': '))
        tk.messagebox.showinfo("Saved!", f"The image saved path: {self.saved_image_path}")

    def get_dir_img_names(self, bowl_idx, img_idx):
        """
        helper function to get current working bowl dir and
        current working image name
        :return:
        """
        cwd = self.bowl_dirs[bowl_idx]
        img_name = os.path.split(self.bowl_images[img_idx])[-1]
        return cwd, img_name

    def save_current_control_point_state(self):
        if self.control_point_tracker.len() > 0:
            # new polygon is created
            # when getting here, [self.bowl_idx][self.image_idx] should be in the system
            dp = self.control_point_tracker.copy()
            self.control_point_tracker_system[self.bowl_idx][self.image_idx][0].append(dp)

        self.control_point_tracker = DoublyLinkedPoints()
        self.changed_control_points = False

    def reset_canvas(self):
        """
        reset all object in canvas, prepare the canvas for the next image
        :return:
        """
        self.control_point_tracker_system[self.bowl_idx][self.image_idx][1].append(self.line_to_tag.copy())
        self.control_point_tracker_system[self.bowl_idx][self.image_idx][2].append(self.point_to_tag.copy())
        self.save_current_control_point_state()
        all_line_to_tag = self.control_point_tracker_system[self.bowl_idx][self.image_idx][1]
        all_point_to_tag = self.control_point_tracker_system[self.bowl_idx][self.image_idx][2]

        for line_to_tag in all_line_to_tag:
            for _, v in line_to_tag.items():
                self.canvas.delete(v)
        for point_to_tag in all_point_to_tag:
            for _, v in point_to_tag.items():
                self.canvas.delete(v)

        self.line_to_tag = {}
        self.point_to_tag = {}

class MainWindow(ttk.Frame):
    """ Main window class """
    def __init__(self, mainframe, path, output_path):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Advanced Zoom v3.0')
        self.master.geometry('800x800')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)
        canvas = CanvasImage(self.master, path, output_path)  # create widget
        canvas.grid(row=0, column=0)  # show widget

this_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_dir)
filename = 'data'  # place path to your image here
output_path = 'testing/'

app = MainWindow(tk.Tk(), path=filename, output_path=output_path)
app.mainloop()
