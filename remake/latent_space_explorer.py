import tkinter as tk
import tkmacosx as tkm
import tensorflow as tf
import numpy as np
import os
import svgwrite
import rdp
import json

import utils
import sketch_rnn_train
import model

EPSILON = 0.1
MODEL_DIR = '/tmp/sketch_rnn/models/owl/lstm'


class Stack:
    def __init__(self):
        self.__stack = []

    def pop(self):
        if len(self.__stack) > 0:
            return self.__stack.pop()
        return None

    def getLength(self):
        return len(self.__stack)

    def push(self, element):
        self.__stack.append(element)


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
    def strokes_to_svg(sketch, filename='picture.svg', size_coefficient=0.2):
        # Given a sketch in 3-stroke format, draw svg to filename.
        # Adapted from https://github.com/magenta/magenta-demos/blob/main/jupyter-notebooks/Sketch_RNN.ipynb
        # in order to draw svg as series of paths for each stroke,
        # rather than a single large path such that one may include stroke
        # extensions such as colour more easily.

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

    @staticmethod
    def selectiveRDP(sketch, epsilon):
        # Use rdp to simplify a drawing, but specifically protect sections
        # which correspond to 'pen up movements'.
        new_sketch = [sketch[0]]
        last_index = 1
        for i, stroke in enumerate(sketch[1:]):
            if stroke[2]:
                simplified_section = rdp.rdp(
                    sketch[last_index:i+1][:-1],
                    epsilon=epsilon
                )
                for simplified_stroke in simplified_section:
                    new_sketch.append(simplified_stroke)
                new_sketch[-1][-1] = 1
                last_index = i+2
        simplified_section = rdp.rdp(
            sketch[last_index:][:-1],
            epsilon=epsilon
        )
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

    @staticmethod
    def stroke_3_to_stroke_5(sketch, max_len):
        new_sketch = [[0, 0, 1, 0, 0]]
        for stroke in sketch:
            new_stroke = [
                stroke[0], stroke[1], not stroke[2], stroke[2], 0
            ]
            new_sketch.append(new_stroke)
        while len(new_sketch) < max_len:
            new_sketch.append([0, 0, 0, 0, 1])
        return new_sketch


class ModelFactory:
    def __init__(self):
        sketch_rnn_train.download_pretrained_models()
        self.__encode_model, self.__decode_model = self.__load_model(MODEL_DIR)
        self.__session = tf.compat.v1.InteractiveSession()
        self.__session.run(tf.compat.v1.global_variables_initializer())
        self.__load_checkpoint(self.__session, MODEL_DIR)

    def getSession(self):
        return self.__session

    def getEncodeModel(self):
        return self.__encode_model

    def getDecodeModel(self):
        return self.__decode_model

    def __load_checkpoint(self, sess, checkpoint_path):
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

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

        sketch_rnn_train.reset_graph()

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

    def sketchToLatent(self, sketch):
        strokes = DataUtilities.stroke_3_to_stroke_5(
            sketch,
            max_len=self.__modelFactory.getEncodeModel().hps.max_seq_len
        )

        seq_len = [len(sketch)]

        z = self.__modelFactory.getSession().run(
            self.__modelFactory.getEncodeModel().batch_z,
            feed_dict={
                self.__modelFactory.getEncodeModel().input_data: [strokes],
                self.__modelFactory.getEncodeModel().sequence_lengths: seq_len
            }
        )

        return z

    def generateFromLatent(self, z, existing_strokes=None):
        strokes, _ = model.sample(
            self.__modelFactory.getSession(),
            self.__modelFactory.getDecodeModel(),
            seq_len=self.__modelFactory.getEncodeModel().hps.max_seq_len,
            temperature=0.1,
            z=z, greedy_mode=True, existing_strokes=existing_strokes)
        return strokes

    def getLatentSize(self):
        return self.__modelFactory.getDecodeModel().hps.z_size


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


class ExplorationWindow:
    def __init__(self, model_handler, z=None, existing_strokes=None):
        # Erroneous attributes
        # We expect either a z (if the window is for sketch reinterpretation)
        # or some existing strokes (for sketch completion)
        assert (
            (z is None and existing_strokes is not None) or
            (z is not None and existing_strokes is None)
        )
        if z is None:
            # Completion
            self.__z = np.random.normal(
                size=(1, model_handler.getLatentSize())
            )
            self.__existing_strokes = existing_strokes
        else:
            # Reinterpretation
            self.__z = z
            self.__existing_strokes = None

        self.__deviations = {
            'Low': 0.2,
            'Medium': 0.5,
            'High': 0.75}
        self.__model_handler = model_handler
        self.__padding = 15
        self.__previous_zs = Stack()

        # Main window properties
        self.__root = tk.Tk()
        self.__root.minsize(500, 500)

        # Widget creation
        self.__initialiseCanvases()
        self.__saveButton = tk.Button(
            self.__root,
            text='Save Sketch',
            command=self.__saveFavourite
        )
        self.__saveButton.grid(row=3, column=0)
        self.__varianceOptions = ['Low', 'Medium', 'High']
        self.__variance = tk.StringVar(self.__root)
        self.__variance.set('Low')
        self.__varianceMenu = tk.OptionMenu(
            self.__root,
            self.__variance,
            *self.__varianceOptions
        )
        self.__varianceMenu.grid(row=3, column=1)
        self.__returnButton = tk.Button(
            self.__root,
            text='Return to Previous Grid',
            command=self.__return
        )
        self.__returnButton.grid(row=3, column=2)

        self.__root.columnconfigure([i for i in range(3)], weight=1)
        self.__root.rowconfigure([i for i in range(4)], weight=1)
        self.__root.rowconfigure(3, minsize=30)

        self.__deviateLatents()
        self.__updateSketches()

        self.__root.mainloop()

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

        self.__canvases[4].config(highlightbackground='red')

        for i in range(3):
            for j in range(3):
                self.__canvases[i*3+j].grid(row=i, column=j, sticky='nesw')
                self.__canvases[i*3+j].bind(
                    '<Button-1>',
                    lambda event, i=i, j=j: self.__selectCanvas(i*3+j)
                )

    def __selectCanvas(self, position):
        self.__previous_zs.push(self.__zs)
        self.__z = [self.__zs[position]]
        self.__deviateLatents()
        self.__updateSketches()

    def __drawSketch(self, sketch, canvas_index):
        canvas = self.__canvases[canvas_index]
        canvas.delete('all')
        canvas_width = canvas.winfo_reqwidth()-self.__padding
        canvas_height = canvas.winfo_reqheight()-self.__padding
        min_x, max_x, min_y, max_y = DataUtilities.get_bounds(sketch, factor=1)
        sketch_width = max_x-min_x
        sketch_height = max_y-min_y
        w_factor = canvas_width/(sketch_width*2)
        h_factor = canvas_height/(sketch_height*2)
        cursor = (canvas_width/2, canvas_height/2)
        pen_up = 1
        for stroke in sketch:
            new_cursor = (
                cursor[0]+(stroke[0]*w_factor),
                cursor[1]+(stroke[1]*h_factor)
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
        sketches = []
        for z in self.__zs:
            if self.__existing_strokes is not None:
                sketch = self.__model_handler.generateFromLatent(
                    [z],
                    np.array(self.__existing_strokes)
                )
            else:
                sketch = self.__model_handler.generateFromLatent([z])
            sketch = utils.to_normal_strokes(sketch)
            sketches.append(sketch)

        for i in range(9):
            self.__drawSketch(sketches[i], canvas_index=i)

    def __deviateLatents(self):
        z = self.__z[0]
        self.__zs = []
        for i in range(9):
            dev = np.random.normal(
                scale=self.__deviations[self.__variance.get()],
                size=z.shape
            )
            self.__zs.append(z + dev)
        self.__zs[4] = z

    def __saveFavourite(self):
        sketch = self.__model_handler.generateFromLatent([self.__z[0]])
        sketch = utils.to_normal_strokes(sketch)
        DataUtilities.strokes_to_svg(sketch)

    def __return(self):
        if self.__previous_zs.getLength() == 0:
            print('No space to go back!')
        else:
            self.__zs = self.__previous_zs.pop()
            self.__updateSketches()


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
        self.__root.minsize(615, 350)

        # Canvas to draw on
        self.__sketchCanvas = ScalingCanvas(
            self,
            self.__root,
            parent_sketchable=True,
            width=50,
            height=50
        )
        self.__sketchCanvas.config(
            bg='white',
            highlightbackground='black'
        )
        self.__sketchCanvas.grid(
            row=0,
            column=0,
            columnspan=5,
            sticky='nesw'
        )
        self.__sketchCanvas.bind('<B1-Motion>', self.__pen_down)
        self.__sketchCanvas.bind('<ButtonRelease-1>', self.__pen_up)

        # Buttons to initiate sketch generation or completion
        self.__generateButtonFrame = tk.Frame(
            self.__root,
            highlightbackground='black',
            highlightthickness=2
        )
        self.__generateButtonFrame.grid(row=1, column=0, sticky='nesw')
        self.__generateButtonFrame.columnconfigure(0, weight=1)
        self.__generateButtonFrame.rowconfigure(0, weight=1)
        self.__generateButton = tkm.Button(
            self.__generateButtonFrame,
            text='Generate Similar Sketches',
            command=self.__generateExplorationWindow,
        )
        self.__generateButton.bind(
            '<Enter>',
            lambda _: self.__generateButton.config(bg='gray58')
        )
        self.__generateButton.bind(
            '<Leave>',
            lambda _: self.__generateButton.config(bg='white')
        )
        self.__generateButton.grid(row=0, column=0)

        self.__completeButtonFrame = tk.Frame(
            self.__root,
            highlightbackground='black',
            highlightthickness=2
        )
        self.__completeButtonFrame.grid(
            row=1,
            column=4,
            columnspan=1,
            sticky='nesw'
        )
        self.__completeButtonFrame.columnconfigure(0, weight=1)
        self.__completeButtonFrame.rowconfigure(0, weight=1)
        self.__completeButton = tkm.Button(
            self.__completeButtonFrame,
            text='Generate Sketch Completions',
            command=self.__generateCompletionWindow
        )
        self.__completeButton.bind(
            '<Enter>',
            lambda _: self.__completeButton.config(bg='gray58')
        )
        self.__completeButton.bind(
            '<Leave>',
            lambda _: self.__completeButton.config(bg='white')
        )
        self.__completeButton.grid(row=0, column=0)

        # Stroke scale and colour inputs
        self.__optionsFrame = tk.Frame(
            self.__root,
            highlightbackground='black',
            highlightthickness=2
        )
        self.__optionsFrame.grid(
            row=1,
            column=1,
            columnspan=3,
            sticky='nesw'
        )
        self.__sizeSlider = tk.Scale(
            self.__optionsFrame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL
        )
        self.__sizeSlider.bind(
            '<ButtonRelease-1>',
            self.__updateBrushThickness
        )
        self.__sizeSlider.grid(
            row=0,
            column=1,
            pady=(0, 10)
        )
        self.__sizeLabel = tk.Label(
            self.__optionsFrame,
            text='Stroke Size:'
        )
        self.__sizeLabel.grid(
            row=0,
            column=0,
            padx=(0, 10)
        )
        self.__changeColourButton = tk.Button(
            self.__optionsFrame,
            text='Change Brush Colour',
            command=self.__changeColour
        )
        self.__changeColourButton.grid(row=1, column=0, columnspan=2)
        self.__optionsFrame.columnconfigure((0, 1), weight=1)
        self.__optionsFrame.rowconfigure((0, 1), weight=1)

        # Make sure grid cells scale properly with window resize
        self.__root.columnconfigure([i for i in range(3)], weight=1)
        self.__root.rowconfigure(0, weight=1)

        self.__root.mainloop()

    def getBrushThickness(self):
        return self.__brushThickness

    def __updateBrushThickness(self, event):
        self.__lineThickness = self.__sizeSlider.get()

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
            event.y/self.__x_factor,
            0
        ])

    def __paint_line(self, event):
        self.__sketchCanvas.create_line(
            self.__cursor[0],
            self.__cursor[1],
            event.x,
            event.y,
            width=self.__lineThickness
        )

    def __pen_up(self, event):
        if len(self.__sketch) > 0:
            self.__sketch[-1][-1] = 1
            self.__cursor = None

    def __changeColour(self):
        print('g')

    def __generateExplorationWindow(self):
        simplified_sketch_points = DataUtilities.simplify_as_possible(
            self.__sketch,
            self.__modelHandler.getMaxSeqLen()
        )
        simplified_sketch_strokes = DataUtilities.convertPointsToStrokes(
            simplified_sketch_points,
            factor=10
        )

        sketch = DataUtilities.normalise(simplified_sketch_strokes)
        z = self.__modelHandler.sketchToLatent(sketch)
        ExplorationWindow(
            model_handler=self.__modelHandler,
            z=z
        )

    def __generateCompletionWindow(self):
        simplified_sketch_points = DataUtilities.simplify_as_possible(
            self.__sketch,
            self.__modelHandler.getMaxSeqLen()/2
        )
        simplified_sketch_strokes = DataUtilities.convertPointsToStrokes(
            simplified_sketch_points,
            factor=10
        )

        sketch = DataUtilities.normalise(simplified_sketch_strokes)
        sketch = DataUtilities.stroke_3_to_stroke_5(
            sketch,
            self.__modelHandler.getMaxSeqLen()
        )
        ExplorationWindow(
            model_handler=self.__modelHandler,
            existing_strokes=sketch
        )


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    sketch_window = Sketch_Window()
