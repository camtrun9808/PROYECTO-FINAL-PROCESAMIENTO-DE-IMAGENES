import cv2
import numpy as np
import PySimpleGUI as sg
import os.path
import img2pdf


class Trapezoid:

    def __init__(self, img_size, r=5, color=(143, 149, 47),
                 bottom_color=(196, 196, 196), border_color=(255, 0, 135)):
        # Initialize the contours
        self.contours = np.array([[[0 + r, 0 + r]],
                                  [[0 + r, img_size[1] - r]],
                                  [[img_size[0] - r, img_size[1] - r]],
                                  [[img_size[0] - r, 0 + r]]])

        # Initialize the radius of the borders
        self.r = r
        # Initialize the colors of the trapezoid
        self.color = color
        self.bottom_color = bottom_color
        self.border_color = border_color

    def get_border_index(self, coord):
        # A border is return if the coordinates are in its radius
        for i, b in enumerate(self.contours[:, 0, :]):
            dist = sum([(b[i] - x) ** 2 for i, x in enumerate(coord)]) ** 0.5
            if dist < self.r:
                return i
        # If no border, return None
        return None

    def set_border(self, border_index, coord):
        self.contours[border_index, 0, :] = coord


class Scanner():

    def __init__(self, input_path):  # , output_path):
        self.input = input_path.copy()
        #self.output_path = output_path
        self.img_output = None
        # get the shape and size of the input
        self.shape = self.input.shape[:-1]
        self.size = tuple(list(self.shape)[::-1])

        # create a trapezoid to drag and drop and its perspective matrix
        self.M = None
        self.trapezoid = Trapezoid(self.size,
                                   r=min(self.shape) // 100 + 2,
                                   color=(153, 153, 153),
                                   border_color=(255, 0, 136),
                                   bottom_color=(143, 149, 47))

        # Initialize the opencv window

        # to remember wich border is dragged if exists
        self.border_dragged = None

    def draw_trapezoid(self, img):
        # draw the contours of the trapezoid
        cv2.drawContours(img, [self.trapezoid.contours], -1,
                         self.trapezoid.color, self.trapezoid.r // 3)
        # draw its bottom
        cv2.drawContours(img, [self.trapezoid.contours[1:3]], -1,
                         self.trapezoid.bottom_color, self.trapezoid.r // 3)
        # Draw the border of the trapezoid as circles
        for x, y in self.trapezoid.contours[:, 0, :]:
            cv2.circle(img, (x, y), self.trapezoid.r,
                       self.trapezoid.border_color, cv2.FILLED)
        return img

    def drag_and_drop_border(self, event, x, y, flags, param):
        # If the left click is pressed, get the border to drag
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the selected border if exists
            self.border_dragged = self.trapezoid.get_border_index((x, y))

        # If the mouse is moving while dragging a border, set its new positionAxel THEVENOT
        elif event == cv2.EVENT_MOUSEMOVE and self.border_dragged is not None:
            self.trapezoid.set_border(self.border_dragged, (x, y))

        # If the left click is released
        elif event == cv2.EVENT_LBUTTONUP:
            # Remove from memory the selected border
            self.border_dragged = None

    def actualize_perspective_matrices(self):
        # get the source points (trapezoid)
        src_pts = self.trapezoid.contours[:, 0].astype(np.float32)

        # set the destination points to have the perspective output image
        h, w = self.shape
        dst_pts = np.array([[0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]], dtype="float32")

        # compute the perspective transform matrices
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def run(self):
        cv2.namedWindow('Rendering', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Rendering', self.drag_and_drop_border)
        while True:
            self.actualize_perspective_matrices()

            # get the output image according to the perspective transformation
            img_output = cv2.warpPerspective(self.input, self.M, self.size)

            # draw current state of the trapezoid
            img_input = self.draw_trapezoid(self.input.copy())
            # Display until the 'Enter' key is pressed
            cv2.imshow('Rendering', img_input)
            if cv2.waitKey(1) & 0xFF == 13:
                break
        self.img_output = img_output
        # Save the image and exit the process
        #cv2.imwrite(self.output_path, img_output)
        cv2.destroyAllWindows()


"""
1. Crear GUI
2. Detectar si el tamaño cabe en la pantalla
3. escalizar
4. preguntar si  quiere filtrar
5. preproceso(mediana-gaussiana)
6. esquinas
7. transformacion
8. preguntara si quiere umbralizar
9. ummbralizacion
10. guardar
"""
'''
filename = sg.popup_get_file('Enter the file you wish to process')
sg.popup('You entered', filename)

# input = 'entrada.jpeg'
output = 'salida.png'
scanner = Scanner(filename, output)
scanner.run()
image = cv2.imread(input)
print(image.shape)
image2 = cv2.imread(output)
print(image2.shape)
'''

"""
GUI Pagina 1: Seleccionar imagen
"""
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(filename='', key="-IMAGE-")]
]
pagina1 = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ],
    [
        sg.Button('Select'),
        sg.Text("Error. No image selected from the list.",
                text_color="red", key="error_p1", visible=False)
    ]
]

"""
GUI pagina 2: Mostrar tamaño y preguntar si quiere escalizar la imagen
"""
pagina2 = [
    [
        sg.Text("The image selected is:"),
        sg.Text(size=(20, 1), key="Size_selected")
    ],
    [
        sg.Text("Do you want to scale the image for better visualization?")
    ],
    [
        sg.Text("(If a dimension is larger than 600 select \"Scale\" option)"),
        sg.Button('Scale'),
        sg.Button("Don't scale"),
    ],
    [
        sg.Button('<- Prev (Select Image)')
    ]
]

"""
GUI pagina 3: escalización de la imagen
"""
p3c1 = [
    [
        sg.Text("Adjust image size (original size is:"),
        sg.Text(size=(30, 1), key="Size_original"),
    ],
    [
        sg.Slider(range=(1, 100), orientation='h', size=(40, 15), default_value=10,
                  visible=True, key="scale_sld", enable_events=True),
    ],
    [
        sg.Text(size=(20, 1), key="scale_max", visible=False)
        # sg.Button("Scale Image", bind_return_key=True),
    ]
]
p3c2 = [
    [
        sg.Text("Scaled image:")
    ],
    [
        sg.Image(filename='', key='ScaleImg')
    ]
]
pagina3 = [
    [
        sg.Column(p3c1),
        sg.VSeparator(),
        sg.Column(p3c2)
    ],
    [
        sg.Button("Finish scaling"),
        sg.Text("Image is still larger than 600 in at least one dimension",
                visible=False, text_color="red", key="error_scale")
    ]
]


"""
GUI pagina 4: Preguntar si quiere hacer el preproceso a la imagen
"""
pagina4 = [
    [
        sg.Text("Do you want to pre-process (filter) the image?")
    ],
    [
        sg.Button('Filter'),
        sg.Button("Don't filter")
    ],
    [
        sg.Button('<- Prev (Adjust image size)')
    ]
]

"""
Gui pagina5: Filtrado de la imagen
"""
col1p5 = [
    [
        sg.Text("Image preprocessing:"),
        sg.Text("Select filter and use sliders to change filter parameters")
    ],
    [
        sg.Radio("Averaging", "prep-filters", size=(12, 1),
                 default=True, key="Avg", enable_events=True),
        sg.Radio("Median", "prep-filters", size=(12, 1), key="Median", enable_events=True),
        sg.Radio("Gaussian", "prep-filters", size=(12, 1), key="Gauss", enable_events=True),
        sg.Radio("Bilateral", "prep-filters", size=(12, 1), key="Bilat", enable_events=True),
    ],
    [
        sg.Text("Kernel Size", visible=True, key="Kernel_avg"),
        sg.Slider(range=(3, 11), orientation='h', size=(40, 15), default_value=3,
                  visible=True, key="filter_sld1", enable_events=True),
    ],
    [
        sg.Text("K Size", visible=False, key="Kernel_med"),
        sg.Slider(range=(3, 11), orientation='h', size=(40, 15), default_value=3,
                  visible=False, key="filter_sld2", enable_events=True)
    ],
    [
        sg.Text("Error, kernel size must be odd", visible=False, key="error_med", text_color='red')
    ],
    [
        sg.Text("Kernel Size", visible=False, key="Kernel_gauss"),
        sg.Slider(range=(3, 11), orientation='h', size=(40, 15), default_value=3,
                  visible=False, key="filter_sld3", enable_events=True),
    ],
    [
        sg.Text("Error, kernel size must be odd", visible=False,
                key="error_gauss", text_color='red')
    ],
    [
        sg.Text("Diameter(d)", visible=False, key="d_bilat"),
        sg.Slider(range=(1, 10), orientation='h', size=(40, 15), default_value=5,
                  visible=False, key="filter_sld4", enable_events=True),
    ],

    [
        sg.Text("Recommended d<5", visible=False, key="d_bilat2")
    ],
    [
        sg.Text("Sigma", visible=False, key="sigma_bilat"),
        sg.Slider(range=(1, 200), orientation='h', size=(40, 15), default_value=10,
                  visible=False, key="filter_sld5", enable_events=True),
    ],
    [
        sg.Button("Apply filter", key="apply")
    ]

]
col2p5 = [
    [sg.Text("Filtered image:")],
    [sg.Image(filename='', key="Prep-image")],
]
pagina5 = [
    [
        sg.Column(col1p5),
        sg.VSeperator(),
        sg.Column(col2p5),
    ]

]

"""
GUI pagina 6: Seleccion de las esquinas
"""
col1p6 = [
    [
        sg.Text("Select document corners:")
    ],
    [
        sg.Text("A new window will pop up. Click and Drag the corners to fit the document.\n(Keep the green line at the bottom)")
    ],
    [sg.Button("Ok, select corners", key="ok1")],
    [sg.Button("<- Prev (Filter image)", key="prev_to_filter")]
]
col2p6 = [
    [
        sg.Text("Result image", visible=False, key="result_scan")
    ],
    [
        sg.Image(filename='', key="scan_result", visible=False)
    ],
    [sg.Button("Select corners again", visible=False, key="corners")],
    [sg.Button("Done", visible=False, key="finish_scan")]
]
pagina6 = [
    [sg.Column(col1p6),
     sg.VSeparator(),
     sg.Column(col2p6)],
]

"""
GUI pagina 7: Preguntar si quiere umbralizar la imagen
"""
pagina7 = [
    [sg.Text("Threshold:")],
    [sg.Text("Do you want to apply thresholding to the image?")],
    [
        sg.Button("No, keep color", key="no_umbral"),
        sg.Button("Yes, apply threshold", key="umbral"),
        sg.Button("<- Prev (Select Corners)", key="prev_pag6")
    ]
]

"""
GUI pagina 8: Umbralizacion de la imagen
"""
p8_col1sub1 = [
    [
        sg.Radio("Simple", "thres1", size=(12, 1),
                 default=True, key="fix", enable_events=True),
    ],
    [
        sg.Text("Select threshold value with the slider")
    ],
    [
        sg.Slider(range=(1, 255), default_value=127, orientation='h',
                  size=(20, 15), enable_events=True, key="thres_sld1")
    ],

]
p8_col2sub1 = [
    [
        sg.Radio("Adaptative", "thres1", size=(12, 1), key="adapt", enable_events=True)
    ],
    [
        sg.Radio("Mean", "thres2", size=(12, 1), key="adapt_mean",
                 enable_events=True),
        sg.Radio("Gaussian", "thres2", size=(12, 1), key="adapt_gauss",
                 enable_events=True)
    ],
    [
        sg.Text("Block size"),
        sg.Slider(range=(3, 21), default_value=127, orientation='h',
                  size=(20, 15), enable_events=True, key="thres_sld2")
    ],
    [
        sg.Text("Error, the value must be an odd value",
                visible=False, text_color='red', key='error_umbral')
    ],
    [
        sg.Text("C"),
        sg.Slider(range=(-10, 10), orientation='h', size=(20, 15), default_value=0,
                  key="thres_sld3", enable_events=True)
    ]
]
pag8col1 = [
    [
        sg.Text("Threshold:")
    ],
    [
        sg.Text("Select threshold type, change values with the sliders")
    ],
    [
        sg.Column(p8_col1sub1),
        sg.VSeparator(),
        sg.Column(p8_col2sub1)
    ],
]
pag8col2 = [
    [
        sg.Image(filename='', key='img_umbral')
    ],
    [
        sg.Button("Apply Threshold")
    ]
]
pagina8 = [
    [
        sg.Column(pag8col1),
        sg.VSeparator(),
        sg.Column(pag8col2)
    ]
]

"""
GUI pagina 9: Como quiere guardar el documento
"""
pagina9 = [
    [
        sg.Text("Save As:")
    ],
    [
        sg.Input("output file name", key="out_name")
    ],
    [
        sg.Radio(".png", "save", size=(12, 1),
                 default=True, key="png"),
        sg.Radio(".jpeg", "save", size=(12, 1), key="jpeg"),
        sg.Radio(".bmp", "save", size=(12, 1), key="bmp"),
        sg.Radio(".pdf", "save", size=(12, 1), key="pdf"),
    ],
    [
        sg.Button("Save", bind_return_key=True)
    ]
]


"""
GUI layout
"""
layout = [
    [
        sg.Column(pagina1, visible=True, key='-COL1-'),
        sg.Column(pagina2, visible=False, key='-COL2-'),
        sg.Column(pagina3, visible=False, key='-COL3-'),
        sg.Column(pagina4, visible=False, key='-COL4-'),
        sg.Column(pagina5, visible=False, key='-COL5-'),
        sg.Column(pagina6, visible=False, key='-COL6-'),
        sg.Column(pagina7, visible=False, key='-COL7-'),
        sg.Column(pagina8, visible=False, key='-COL8-'),
        sg.Column(pagina9, visible=False, key='-COL9-'),
    ],
    [
        sg.Button('Exit'),
    ]

]

window = sg.Window("Scan", layout)
layout = 1
fix = True

"""
GUI Loop
"""
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    """
    Eventos pagina 1:
    """
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpeg", ".gif", ".bmp"))
        ]
        window["-FILE LIST-"].update(fnames)
    if event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            file = cv2.imread(filename)
            w_file = file.shape[1]
            h_file = file.shape[0]
            if (w_file or h_file > 500):
                if w_file/h_file == 5/4:
                    size_view = (500, 400)
                elif h_file/w_file == 5/4:
                    size_view = (400, 500)
                elif w_file/h_file == 4/3:
                    size_view = (400, 300)
                elif h_file/w_file == 4/3:
                    size_view = (300, 400)
                elif w_file/h_file == 3/2:
                    size_view = (300, 200)
                elif h_file/w_file == 3/2:
                    size_view = (200, 300)
                elif w_file/h_file == 8/5:
                    size_view = (400, 250)
                elif h_file/w_file == 8/5:
                    size_view = (250, 400)
                elif w_file/h_file == 16/9:
                    size_view = (320, 180)
                elif h_file/w_file == 16/9:
                    size_view = (180, 320)
                elif w_file/h_file == 5/3:
                    size_view = (350, 210)
                elif h_file/w_file == 5/3:
                    size_view = (210, 350)
                elif w_file/h_file == 17/9:
                    size_view = (340, 180)
                elif h_file/w_file == 17/9:
                    size_view = (180, 340)
                elif w_file/h_file == 21/9:
                    size_view = (420, 180)
                elif h_file/w_file == 21/9:
                    size_view = (180, 420)
                else:
                    size_view = (int(w_file*30/100), int(h_file*30/100))
                resized = cv2.resize(file, size_view, interpolation=cv2.INTER_AREA)
            else:
                resized = file
            imgbytes = cv2.imencode('.png', resized)[1].tobytes()
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(data=imgbytes)
        except:
            pass
    if event == "Select" and file is not None:  # Pasa a la pagina 2
        window[f'-COL{layout}-'].update(visible=False)
        layout = 2
        window[f'-COL{layout}-'].update(visible=True)
        # Cuando pasa a la pagina se actualiza el texto del tamaño en pag 2 y 3
        window['Size_selected'].update(str(w_file)+"x"+str(h_file))
        window['Size_original'].update(str(w_file)+"x"+str(h_file)+")")
        window["error_p1"].update(visible=False)
    if event == "Select" and file is None:
        window["error_p1"].update(visible=True)

    """
    Eventos pagina 2
    """
    if event == "<- Prev (Select Image)":  # Se devuelve a la primera pagina
        window[f'-COL{layout}-'].update(visible=False)
        layout = 1
        window[f'-COL{layout}-'].update(visible=True)
    if event == "Scale":  # Guarda que quiere escalizar y avanza a la pag 3
        scale = True
        window[f'-COL{layout}-'].update(visible=False)
        layout = 3
        window[f'-COL{layout}-'].update(visible=True)
        if w_file > 600 or h_file > 600:
            w_scale = w_file/10
            h_scale = h_file/10
            scaled_img = cv2.resize(file, (int(w_scale), int(h_scale)),
                                    interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', scaled_img)[1].tobytes()
            window["ScaleImg"].update(data=imgbytes)
        else:
            w_scale = w_file
            h_scale = h_file
            scaled_img = file.copy()
            imgbytes = cv2.imencode('.png', scaled_img)[1].tobytes()
            window["ScaleImg"].update(data=imgbytes)
    if (event == "Don't scale" and (w_file <= 600 or h_file <= 600)):
        # Guarda que no escaliza y avanza pag3
        scale = False
        window[f'-COL{layout}-'].update(visible=False)
        layout = 4
        window[f'-COL{layout}-'].update(visible=True)
        scaled_img = file.copy()
        w_scale = w_file
        h_scale = h_file
        window["error_scale"].update(visible=False)
    if (event == "Don't scale" and (w_file > 600 or h_file > 600)):
        window["error_scale"].update(visible=True)

    """
    Eventos pagina 3
    """
    if event == "Finish scaling" and (w_scaler > 600 or h_scaler > 600):
        window["error_scale"].update(visible=True)
    if event == "Finish scaling" and (w_scaler <= 600 or h_scaler <= 600):
        window["error_scale"].update(visible=False)
        window[f'-COL{layout}-'].update(visible=False)
        layout = 4
        window[f'-COL{layout}-'].update(visible=True)
        w_scale = w_scaler
        h_scale = h_scaler
    if event == "scale_sld":
        scaler = int(values['scale_sld'])
        w_scaler = int((w_file*scaler)/100)
        h_scaler = int((h_file*scaler)/100)
        window["ScaleImg"].update(data=None)
        if (w_scaler > 600) or (h_scaler > 600):
            window["scale_max"].update(visible=True)
            window["scale_max"].update(
                "Maximum size for correct visualization is 600 in both dimensions, scaled image is:"+str(w_scaler)+"x"+str(h_scaler))
            window["scale_max"].update(text_color="red")
        if (w_scaler <= 600 or h_scaler <= 600):
            window["scale_max"].update(visible=False)
            scaled_img = cv2.resize(file, (w_scaler, h_scaler), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', scaled_img)[1].tobytes()
            window["ScaleImg"].update(data=imgbytes)

    """
    Eventos pagina 4
    """
    if event == "<- Prev (Adjust image size)":  # se devuelve a la pag 2
        window[f'-COL{layout}-'].update(visible=False)
        layout = 2
        window[f'-COL{layout}-'].update(visible=True)
    if event == "Filter":  # selecciona que filtra y pasa a la pagina 4
        window[f'-COL{layout}-'].update(visible=False)
        layout = 5
        ftype = "avg"
        errorP5 = False
        window[f'-COL{layout}-'].update(visible=True)
        filtered = cv2.blur(scaled_img, (3, 3))
        imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
        window["Prep-image"].update(data=imgbytes)
    if event == "Don't filter":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 6
        filtered = scaled_img.copy()
        window[f'-COL{layout}-'].update(visible=True)
        scanner = Scanner(filtered)  # Llama a la clase scanner y le pasa la imagen

    """
    Eventos pagina 5
    """
    if event == "apply":
        if not errorP5:
            window[f'-COL{layout}-'].update(visible=False)
            layout = 6
            window[f'-COL{layout}-'].update(visible=True)
            scanner = Scanner(filtered)  # Llama a la clase scanner y le pasa la imagen
    if event == "Avg":
        window["Kernel_avg"].update(visible=True)
        window["filter_sld1"].update(visible=True)
        window["Kernel_med"].update(visible=False)
        window["filter_sld2"].update(visible=False)
        window["Kernel_gauss"].update(visible=False)
        window["filter_sld3"].update(visible=False)
        window["d_bilat"].update(visible=False)
        window["filter_sld4"].update(visible=False)
        window["d_bilat2"].update(visible=False)
        window["sigma_bilat"].update(visible=False)
        window["filter_sld5"].update(visible=False)
        if ftype != "avg":
            filtered = cv2.blur(scaled_img, (3, 3))
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
        ftype = "avg"
    if event == "Median":
        window["Kernel_avg"].update(visible=False)
        window["filter_sld1"].update(visible=False)
        window["Kernel_med"].update(visible=True)
        window["filter_sld2"].update(visible=True)
        window["Kernel_gauss"].update(visible=False)
        window["filter_sld3"].update(visible=False)
        window["d_bilat"].update(visible=False)
        window["filter_sld4"].update(visible=False)
        window["d_bilat2"].update(visible=False)
        window["sigma_bilat"].update(visible=False)
        window["filter_sld5"].update(visible=False)
        if ftype != "med":
            filtered = cv2.medianBlur(scaled_img, 3)
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
        ftype = "med"
    if event == "Gauss":
        window["Kernel_avg"].update(visible=False)
        window["filter_sld1"].update(visible=False)
        window["Kernel_med"].update(visible=False)
        window["filter_sld2"].update(visible=False)
        window["Kernel_gauss"].update(visible=True)
        window["filter_sld3"].update(visible=True)
        window["d_bilat"].update(visible=False)
        window["filter_sld4"].update(visible=False)
        window["d_bilat2"].update(visible=False)
        window["sigma_bilat"].update(visible=False)
        window["filter_sld5"].update(visible=False)
        if ftype != "gauss":
            filtered = cv2.GaussianBlur(scaled_img, (3, 3), 0)
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
        ftype = "gauss"
    if event == "Bilat":
        window["Kernel_avg"].update(visible=False)
        window["filter_sld1"].update(visible=False)
        window["Kernel_med"].update(visible=False)
        window["filter_sld2"].update(visible=False)
        window["Kernel_gauss"].update(visible=False)
        window["filter_sld3"].update(visible=False)
        window["d_bilat"].update(visible=True)
        window["filter_sld4"].update(visible=True)
        window["d_bilat2"].update(visible=True)
        window["sigma_bilat"].update(visible=True)
        window["filter_sld5"].update(visible=True)
        if ftype != "bilat":
            filtered = cv2.bilateralFilter(scaled_img, 5, 10, 10)
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
        ftype = "bilat"
    if event == "filter_sld1":
        ks = int(values['filter_sld1'])
        filtered = cv2.blur(scaled_img, (ks, ks))
        imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
        window["Prep-image"].update(data=imgbytes)
        errorP5 = False
    if event == "filter_sld2":
        window['error_med'].update(visible=False)
        ks = int(values['filter_sld2'])
        errorP5 = False
        if (ks % 2 == 0):
            window["error_med"].update(visible=True)
            errorP5 = True
        else:
            filtered = cv2.medianBlur(scaled_img, ks)
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
    if event == "filter_sld3":
        ks = int(values['filter_sld3'])
        window["error_gauss"].update(visible=False)
        errorP5 = False
        if (ks % 2 == 0):
            window["error_gauss"].update(visible=True)
            errorP5 = True
        else:
            filtered = cv2.GaussianBlur(scaled_img, (ks, ks), 0)
            imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
            window["Prep-image"].update(data=imgbytes)
    if event == "filter_sld4":
        errorP5 = False
        d = int(values['filter_sld4'])
        sigma = int(values['filter_sld5'])
        filtered = cv2.bilateralFilter(scaled_img, d, sigma, sigma)
        imgbytes = cv2.imencode('.png', filtered)[1].tobytes()
        window["Prep-image"].update(data=imgbytes)

    """
    Eventos pagina 6
    """
    if event == "finish_scan":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 7
        scan_gr = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
        scan_gr_scale = cv2.cvtColor(resize_scanned, cv2.COLOR_BGR2GRAY)
        window[f'-COL{layout}-'].update(visible=True)
    if event == ("ok1" or "Select corners again"):
        scanner.run()
        scanned = scanner.img_output.copy()
        resize_scanned = cv2.resize(scanned, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode('.png', resize_scanned)[1].tobytes()
        window["result_scan"].update(visible=True)
        window["scan_result"].update(data=imgbytes, visible=True)
        window["corners"].update(visible=True)
        window["finish_scan"].update(visible=True)
    if event == "prev_to_filter":
        window["result_scan"].update(visible=False)
        window["scan_result"].update(visible=False)
        window["corners"].update(visible=False)
        window["finish_scan"].update(visible=False)
        window[f'-COL{layout}-'].update(visible=False)
        layout = 4
        window[f'-COL{layout}-'].update(visible=True)

    """
    Eventos pagina 7
    """
    if event == "no_umbral":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 9
        window[f'-COL{layout}-'].update(visible=True)
        final_img = scanned.copy()
    if event == "umbral":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 8
        window[f'-COL{layout}-'].update(visible=True)
        ret, scan_bw = cv2.threshold(scan_gr, 127, 255, cv2.THRESH_BINARY)
        scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
        window["img_umbral"].update(data=imgbytes)
    if event == "prev_pag6":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 6
        window[f'-COL{layout}-'].update(visible=True)

    """
    Eventos pagina 8
    """
    if event == "fix":
        window['adapt_mean'].ResetGroup()
        ret, scan_bw = cv2.threshold(scan_gr, 127, 255, cv2.THRESH_BINARY)
        scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
        window['thres_sld1'].update(value=127)
        imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
        window["img_umbral"].update(data=imgbytes)
        fix = True
    if event == "adapt":
        window['adapt_mean'].update(value=True)
        scan_bw = cv2.adaptiveThreshold(
            scan_gr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
        window['thres_sld1'].update(value=127)
        imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
        window["img_umbral"].update(data=imgbytes)
        fix = False
    if event == "adapt_mean":
        window["adapt"].update(value=True)
    if event == "adapt_gauss":
        window["adapt"].update(value=True)
    if event == "thres_sld1":
        if fix:
            ret, scan_bw = cv2.threshold(scan_gr, values['thres_sld1'], 255, cv2.THRESH_BINARY)
            scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
            window["img_umbral"].update(data=imgbytes)
    if event == "thres_sld2":
        if not fix:
            if int(values['thres_sld2']) % 2 == 1:
                window["error_umbral"].update(visible=False)
                scan_bw = cv2.adaptiveThreshold(scan_gr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, int(values['thres_sld2']), int(values['thres_sld3']))
                scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale),
                                           interpolation=cv2.INTER_AREA)
                imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
                window["img_umbral"].update(data=imgbytes)
            else:
                window["error_umbral"].update(visible=True)
    if event == "thres_sld3":
        if not fix:
            scan_bw = cv2.adaptiveThreshold(scan_gr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, int(values['thres_sld2']), int(values['thres_sld3']))
            scan_bw_scale = cv2.resize(scan_bw, (w_scale, h_scale), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', scan_bw_scale)[1].tobytes()
            window["img_umbral"].update(data=imgbytes)
    if event == "Apply Threshold":
        window[f'-COL{layout}-'].update(visible=False)
        layout = 9
        window[f'-COL{layout}-'].update(visible=True)
        final_img = scan_bw.copy()

    """
    Eventos pagina 9
    """
    if event == "Save":
        name_final = str(values['out_name'])
        if window['png'].Get():
            cv2.imwrite(name_final+".png", final_img)
        if window['jpeg'].Get():
            cv2.imwrite(name_final+".jpeg", final_img)
        if window['bmp'].Get():
            cv2.imwrite(name_final+".bmp", final_img)
        if window['pdf'].Get():
            final_img.save(name_final+".pdf")
        break

window.close()
