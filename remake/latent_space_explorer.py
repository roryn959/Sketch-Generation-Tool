import tkinter as tk
import tensorflow as tf
import numpy as np
import os
import svgwrite
import rdp
import json

import utils
import sketch_rnn_train
import model

tf.compat.v1.disable_v2_behavior()
EPSILON = 0.1
MODEL_DIR = '/tmp/sketch_rnn/models/flamingo/lstm_uncond'


class DataUtilities:
    @staticmethod
    def normalise(sketch):
        # Normalises data as in section 1 of sketch-rnn paper appendix

        # First find mean
        meanx = 0
        meany = 0
        for stroke in sketch:
            meanx += stroke[0]
            meany += stroke[1]
        meanx /= len(sketch)
        meany /= len(sketch)

        # Then find variance
        varx = 0
        vary = 0
        for stroke in sketch:
            varx += (stroke[0]-meanx)**2
            vary += (stroke[1]-meany)**2

        varx /= (len(sketch)+1)
        vary /= (len(sketch)+1)

        # Then get standard deviations
        sdx = varx**(1/2)
        sdy = vary**(1/2)

        # Then normalise
        new_sketch = []
        for stroke in sketch:
            new_sketch.append([stroke[0]/sdx, stroke[1]/sdy, stroke[2]])
        
        return new_sketch

    @staticmethod
    def get_bounds(data, factor=10):
        """Return bounds of data."""
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        abs_x = 0
        abs_y = 0
        for i in range(len(data)):
            x = float(data[i][0]) / factor
            y = float(data[i][1]) / factor
            abs_x += x
            abs_y += y
            min_x = min(min_x, abs_x)
            min_y = min(min_y, abs_y)
            max_x = max(max_x, abs_x)
            max_y = max(max_y, abs_y)

        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def strokes_to_svg(sketch, filename='picture', size_coefficient=0.2):
        # Given a sketch in 3-stroke format, draw svg to filename.
        # Adapted from https://github.com/magenta/magenta-demos/blob/main/jupyter-notebooks/Sketch_RNN.ipynb
        # in order to draw svg as series of paths for each stroke, rather than a single large
        # path such that one may stroke extensions such as colour more easily.

        tf.compat.v1.gfile.MakeDirs(os.path.dirname(filename))

        min_x, max_x, min_y, max_y = DataUtilities.get_bounds(sketch, size_coefficient)
        dims = (50 + max_x - min_x, 50 + max_y - min_y)
        dwg = svgwrite.Drawing(filename, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
        x = 25 - min_x
        y = 25 - min_y
        lift_pen = 1

        paths = []
        for i in range(len(sketch)):
            x += float(sketch[i][0])/size_coefficient
            y += float(sketch[i][1])/size_coefficient

            end_of_stroke = sketch[i][2]

            if lift_pen:
                p = f'M{x},{y}'
            else:
                p += f'L{x},{y} '

            if end_of_stroke:
                paths.append(p)

            lift_pen = end_of_stroke

            i += 1

        for path in paths:
            dwg.add(dwg.path(path).stroke('black', 1).fill('none'))

        dwg.save()
        print('Picture saved!')

    @staticmethod
    def selectiveRDP(sketch, epsilon):
        # Use rdp to simplify a drawing, but specifically protect sections
        # which correspond to 'pen up movements'.
        new_sketch = [sketch[0]]
        last_index = 1
        for i, stroke in enumerate(sketch[1:]):
            if stroke[2]:
                simplified_section = rdp.rdp(sketch[last_index:i+1][:-1], epsilon=epsilon)
                for simplified_stroke in simplified_section:
                    new_sketch.append(simplified_stroke)
                new_sketch[-1][-1] = 1
                last_index = i+2
        simplified_section = rdp.rdp(sketch[last_index:][:-1], epsilon=epsilon)
        for simplified_stroke in simplified_section:
            new_sketch.append(simplified_stroke)
        return new_sketch

    @staticmethod
    def simplify_as_possible(sketch, max_seq_len):
        if len(sketch) < max_seq_len:
            return sketch

        epsilon = 0.1
        while True:
            simplified_sketch = DataUtilities.selectiveRDP(sketch, epsilon)
            if len(simplified_sketch) < max_seq_len:
                return simplified_sketch
            epsilon += 0.1

    @staticmethod
    def convertPointsToStrokes(points, factor):
        # Converts a list of points to stroke-3 format
        strokes = [points[0]]
        for i in range(1, len(points)):
            currentPoint, lastPoint = points[i], points[i-1]
            movementx = currentPoint[0]-lastPoint[0]
            movementy = currentPoint[1]-lastPoint[1]
            pen_state = currentPoint[-1]
            strokes.append([movementx*factor, movementy*factor, pen_state])
        return strokes


class ModelFactory:
    def __init__(self):
        sketch_rnn_train.download_pretrained_models()
        self.__encode_model, self.__decode_model = self.__load_model(MODEL_DIR)
        self.__session = tf.compat.v1.InteractiveSession()
        self.__session.run(tf.compat.v1.global_variables_initializer())
        sketch_rnn_train.load_checkpoint(self.__session, MODEL_DIR)

    def getSession(self):
        return self.__session

    def getEncodeModel(self):
        return self.__encode_model

    def getDecodeModel(self):
        return self.__decode_model

    def __load_model(self, model_dir):
        # Code modified from: https://github.com/magenta/magenta-demos/blob/main/jupyter-notebooks/Sketch_RNN.ipynb

        model_params = model.get_default_hparams()
        path = os.path.join(model_dir, 'model_config.json')
        with tf.compat.v1.gfile.Open(path, 'r') as f:
            data = json.load(f)

        fix_list = [
            'conditional',
            'is_training',
            'use_input_dropout',
            'use_output_dropout',
            'use_recurrent_dropout'
        ]

        for fix in fix_list:
            data[fix] = (data[fix] == 1)

        model_params.parse_json(json.dumps(data))

        # *** Try higher batch size ***
        model_params.batch_size = 1
        encode_model_params = model.copy_hparams(model_params)

        # No dropout because we aren't training
        encode_model_params.use_input_dropout = 0
        encode_model_params.use_recurrent_dropout = 0
        encode_model_params.use_output_dropout = 0
        encode_model_params.is_training = 0

        decode_model_params = model.copy_hparams(encode_model_params)
        decode_model_params.max_seq_len = 1

        encode_model = model.Model(
            encode_model_params,
            reuse=tf.compat.v1.AUTO_REUSE
        )

        decode_model = model.Model(
            decode_model_params,
            reuse=tf.compat.v1.AUTO_REUSE
        )

        return encode_model, decode_model


class ModelHandler:
    def __init__(self):
        self.__modelFactory = ModelFactory()

    def getMaxSeqLen(self):
        return self.__modelFactory.getEncodeModel().hps.max_seq_len

    def toLatent(self, sketch):
        # *** Following casts are unnecessary!
        sketch = np.array(sketch)
        strokes = utils.to_big_strokes(
            sketch,
            max_len=self.__modelFactory.getEncodeModel().hps.max_seq_len).tolist()

        # *** Consider adapting conversion method instead ***
        strokes.insert(0, [0, 0, 1, 0, 0])

        seq_len = [len(sketch)]

        z = self.__modelFactory.getSession().run(
            self.__modelFactory.getEncodeModel().batch_z,
            feed_dict={
                self.__modelFactory.getEncodeModel().input_data: [strokes],
                self.__modelFactory.getEncodeModel().sequence_lengths: seq_len
            }
        )

        return z

    def fromLatent(self, z):
        strokes, _ = model.sample(
            self.__modelFactory.getSession(),
            self.__modelFactory.getDecodeModel(),
            seq_len=self.__modelFactory.getEncodeModel().hps.max_seq_len,
            temperature=0.01,
            z=z, greedy_mode=True)
        return strokes


class ScalingCanvas(tk.Canvas):
    def __init__(self, window, parent, parent_sketchable, **kwargs):
        super(ScalingCanvas, self).__init__(parent, **kwargs)
        self.bind('<Configure>', self.__rescale_components)
        self.__height = self.winfo_reqheight()
        self.__width = self.winfo_reqwidth()
        self.__window = window
        self.__parent_sketchable = parent_sketchable

    def __rescale_components(self, event):
        # Find factors by which the canvas was resized
        width_factor = float(event.width)/self.__width
        height_factor = float(event.height)/self.__height

        # We also need do rescale the stroke magnitudes.
        # E.g. a 5 cm stroke will span a whole small canvas,but a small amount
        # of a large one. But the stroke encoding should be the same.
        if self.__parent_sketchable:
            self.__window.adjustStrokeFactors((width_factor, height_factor))

        # Update width and height to new values
        self.__width = event.width
        self.__height = event.height

        # Re-scale all the components inside the canvas
        # args->(tags, offsetX, offsetY, xScale, yScale)
        self.scale('all', 0, 0, width_factor, height_factor)


class ExploreWindow:
    def __init__(self, model_handler, z):
        # Erroneous attributes
        self.__z = z
        print(z)
        self.__model_handler = model_handler
        self.__padding = 5

        # Main window properties
        self.__root = tk.Tk()
        self.__root.minsize(500, 500)

        # Widget creation
        self.__initialiseCanvases()
        self.__saveAndQuitButton = tk.Button(
            self.__root,
            text='Save and Quit'
        )
        self.__saveAndQuitButton.grid(row=3, column=0)
        self.__varianceOptions = ['Low', 'Medium', 'High']
        self.__variance = tk.StringVar(self.__root)
        self.__variance.set('Medium')
        self.__varianceMenu = tk.OptionMenu(
            self.__root,
            self.__variance,
            *self.__varianceOptions
        )
        self.__varianceMenu.grid(row=3, column=1)
        self.__returnButton = tk.Button(
            self.__root,
            text='Return to Previous Grid'
        )
        self.__returnButton.grid(row=3, column=2)

        self.__root.columnconfigure([i for i in range(3)], weight=1)
        self.__root.rowconfigure([i for i in range(4)], weight=1)
        self.__root.rowconfigure(3, minsize=30)

        self.__updateSketches()

        self.__root.mainloop()

    def __findMeanAndVarOfZ(self):
        mean_sum = 0
        for z in self.__z[0]:
            mean_sum += z
        mean = mean_sum/self.__z[0].shape[0]

        sig_sq = 0
        for z in self.__z[0]:
            sig_sq += (z - mean)**2
        sig_sq /= self.__z[0].shape[0]

    def __initialiseCanvases(self):
        self.__canvases = [
            ScalingCanvas(
                self,
                self.__root,
                parent_sketchable=False,
                width=200,
                height=200,
                bg='white',
                highlightbackground='black'
            ) for i in range(9)]

        for i in range(3):
            for j in range(3):
                self.__canvases[i*3+j].grid(row=i, column=j, sticky='nesw')

    def __drawSketch(self, sketch, canvas_index):
        canvas = self.__canvases[canvas_index]
        min_x, max_x, min_y, max_y = DataUtilities.get_bounds(sketch, factor=1)
        canvas_width = canvas.winfo_reqwidth()
        canvas_height = canvas.winfo_reqheight()
        dims = (self.__padding + max_x - min_x, self.__padding + max_y - min_y)
        x_factor = canvas_width/dims[0]
        y_factor = canvas_height/dims[1]
        cursor = (self.__padding*x_factor, self.__padding*y_factor)
        pen_up = 1
        for stroke in sketch:
            new_cursor = (
                cursor[0]+stroke[0]*x_factor,
                cursor[1]+stroke[1]*y_factor
            )
            if not pen_up:
                canvas.create_line(
                    cursor[0],
                    cursor[1],
                    new_cursor[0],
                    new_cursor[1]
                )
            cursor = new_cursor
            pen_up = stroke[2]

    def __updateSketches(self):
        zs = []
        for i in range(9):
            zs.append(self.__z)

        sketches = []
        for z in zs:
            sketch = self.__model_handler.fromLatent(self.__z)
            sketch = utils.to_normal_strokes(sketch)
            sketches.append(sketch)

        DataUtilities.strokes_to_svg(sketches[0])

        for i in range(9):
            self.__drawSketch(sketches[i], canvas_index=i)


class Sketch_Window:
    def __init__(self):
        # Erroneous attributes
        self.__modelHandler = ModelHandler()
        self.__lineThickness = 3
        self.__ovalSize = 2
        self.__sketch = []
        self.__base_window_dims = (700, 600)
        self.__x_factor = 1
        self.__y_factor = 1
        self.__cursor = None

        # Main window properties
        self.__root = tk.Tk()
        self.__root.title('Sketch Pad')
        self.__root.geometry('700x600')
        self.__root.minsize(400, 350)

        # Canvas to draw on
        self.__sketchCanvas = ScalingCanvas(
            self,
            self.__root,
            parent_sketchable=True,
            width=50,
            height=50
        )
        self.__sketchCanvas.config(bg='white', highlightbackground='black')
        self.__sketchCanvas.grid(row=0, column=0, columnspan=4, sticky='nesw')
        self.__sketchCanvas.bind('<B1-Motion>', self.__pen_down)
        self.__sketchCanvas.bind('<ButtonRelease-1>', self.__pen_up)

        # Buttons to initiate sketch generation or completion
        self.__generateButton = tk.Button(
            self.__root,
            text='Generate Similar Sketches',
            command=self.__generateExploreWindow
        )
        self.__finishButton = tk.Button(
            self.__root,
            text='Generate Finished Sketches'
        )
        self.__generateButton.grid(row=1, column=0, columnspan=2, pady=(0, 5))
        self.__finishButton.grid(row=1, column=2, columnspan=2, pady=(0, 5))

        # Make sure grid cells scale properly with window resize
        self.__root.columnconfigure([i for i in range(4)], weight=1)
        self.__root.rowconfigure(0, weight=1)

        self.__root.mainloop()

    def getBrushThickness(self):
        return self.__brushThickness

    def setBrushThickness(self, value):
        self.__brushThickness = value

    def getBaseDims(self):
        return self.__base_window_dims

    def getFactors(self):
        return self.__x_factor, self.__y_factor

    def adjustStrokeFactors(self, factors):
        (x_factor, y_factor) = factors
        self.__x_factor *= x_factor
        self.__y_factor *= y_factor

    def __pen_down(self, event):
        if self.__cursor is not None:
            self.__paint_line(event)

        self.__cursor = (event.x, event.y)

        # Add movement to stroke list after scaling based on canvas size
        self.__sketch.append([
            event.x/self.__x_factor,
            event.y/self.__x_factor, 0
        ])

    def __paint_oval(self, event):
        self.__sketchCanvas.create_oval(
            event.x-self.__ovalSize,
            event.y-self.__ovalSize,
            event.x+self.__ovalSize,
            event.y+self.__ovalSize,
            fill='black',
            outline='black'
        )

    def __paint_line(self, event):
        self.__sketchCanvas.create_line(
            self.__cursor[0],
            self.__cursor[1],
            event.x,
            event.y,
            width=self.__lineThickness
        )

    def __pen_up(self, event):
        self.__sketch[-1][-1] = 1
        self.__cursor = None

    def __generateExploreWindow(self):
        simplified_sketch_points = DataUtilities.simplify_as_possible(
            self.__sketch,
            self.__modelHandler.getMaxSeqLen()
        )
        simplified_sketch_strokes = DataUtilities.convertPointsToStrokes(
            simplified_sketch_points,
            factor=1
        )

        sketch = DataUtilities.normalise(simplified_sketch_strokes)

        DataUtilities.strokes_to_svg(
            sketch,
            'picture')

        z = self.__modelHandler.toLatent(sketch)
        ExploreWindow(model_handler=self.__modelHandler, z=z)


if __name__ == '__main__':
    sketch_window = Sketch_Window()