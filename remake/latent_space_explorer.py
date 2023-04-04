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

EPSILON = 0.1
MODEL_DIR = '/tmp/sketch_rnn/models/owl/lstm'


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

    def toLatent(self, sketch):
        # *** Following casts are unnecessary!
        sketch = np.array(sketch, dtype=np.float32)
        strokes = utils.to_big_strokes(
            sketch,
            max_len=self.__modelFactory.getEncodeModel().hps.max_seq_len
            ).tolist()

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
        self.__deviation = 0.5
        self.__deviateLatents()
        self.__model_handler = model_handler
        self.__padding = 15

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
                self.__canvases[i*3+j].bind('<Button-1>', lambda event, i=i, j=j: self.__selectCanvas(i*3+j))

    def __selectCanvas(self, position):
        print(f'canvas {position} selected')
        self.__z = self.__zs[position]
        self.__updateSketches()

    def __drawSketch(self, sketch, canvas_index):
        canvas = self.__canvases[canvas_index]
        print(f'clearing canvas {canvas_index}')
        canvas.delete('all')
        input()
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
            sketch = self.__model_handler.fromLatent([z])
            sketch = utils.to_normal_strokes(sketch)
            sketches.append(sketch)

        for i in range(9):
            self.__drawSketch(sketches[i], canvas_index=i)

    def __deviateLatents(self):
        z = self.__z[0]
        self.__zs = []
        for i in range(9):
            dev = np.random.normal(scale=self.__deviation, size=z.shape)
            self.__zs.append(z + dev)


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
            factor=10
        )

        sketch = DataUtilities.normalise(simplified_sketch_strokes)

        DataUtilities.strokes_to_svg(
            sketch,
            'picture')

        z = self.__modelHandler.toLatent(sketch)
        ExploreWindow(model_handler=self.__modelHandler, z=z)


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    sketch_window = Sketch_Window()

    # np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    # tf.compat.v1.disable_v2_behavior()

    # data_dir = 'http://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep/'
    # models_root_dir = '/tmp/sketch_rnn/models'
    # model_dir = '/tmp/sketch_rnn/models/aaron_sheep/layer_norm'
    # sketch_rnn_train.download_pretrained_models(models_root_dir=models_root_dir)

    # def load_env_compatible(data_dir, model_dir):
    #     """Loads environment for inference mode, used in jupyter notebook."""
    #     # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    #     # to work with depreciated tf.HParams functionality
    #     model_params = model.get_default_hparams()
    #     with tf.compat.v1.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    #         data = json.load(f)
    #     fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    #     for fix in fix_list:
    #         data[fix] = (data[fix] == 1)
    #     model_params.parse_json(json.dumps(data))
    #     return sketch_rnn_train.load_dataset(data_dir, model_params, inference_mode=True)

    # def load_model_compatible(model_dir):
    #     """Loads model for inference mode, used in jupyter notebook."""
    #     # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    #     # to work with depreciated tf.HParams functionality
    #     model_params = model.get_default_hparams()
    #     with tf.compat.v1.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    #         data = json.load(f)
    #     fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    #     for fix in fix_list:
    #         data[fix] = (data[fix] == 1)
    #     model_params.parse_json(json.dumps(data))
    #     model_params.batch_size = 1  # only sample one at a time
    #     eval_model_params = model.copy_hparams(model_params)
    #     eval_model_params.use_input_dropout = 0
    #     eval_model_params.use_recurrent_dropout = 0
    #     eval_model_params.use_output_dropout = 0
    #     eval_model_params.is_training = 0
    #     sample_model_params = model.copy_hparams(eval_model_params)
    #     sample_model_params.max_seq_len = 1  # sample one point at a time
    #     return [model_params, eval_model_params, sample_model_params]

    # def load_dataset(data_dir, model_params, inference_mode=False):
        # """Loads the .npz file, and splits the set into train/valid/test."""

        # # normalizes the x and y columns using the training set.
        # # applies same scaling factor to valid and test set.

        # if isinstance(model_params.data_set, list):
        #     datasets = model_params.data_set
        # else:
        #     datasets = [model_params.data_set]

        # train_strokes = None
        # valid_strokes = None
        # test_strokes = None

        # for dataset in datasets:
        #     if data_dir.startswith('http://') or data_dir.startswith('https://'):
        #     data_filepath = '/'.join([data_dir, dataset])
        #     tf.compat.v1.logging.info('Downloading %s', data_filepath)
        #     response = requests.get(data_filepath, allow_redirects=True)
        #     data = np.load(BytesIO(response.content), allow_pickle=True, encoding='latin1')
        #     else:
        #     data_filepath = os.path.join(data_dir, dataset)
        #     data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        #     tf.compat.v1.logging.info('Loaded {}/{}/{} from {}'.format(
        #         len(data['train']), len(data['valid']), len(data['test']),
        #         dataset))
        #     if train_strokes is None:
        #     train_strokes = data['train']
        #     valid_strokes = data['valid']
        #     test_strokes = data['test']
        #     else:
        #     train_strokes = np.concatenate((train_strokes, data['train']))
        #     valid_strokes = np.concatenate((valid_strokes, data['valid']))
        #     test_strokes = np.concatenate((test_strokes, data['test']))

        # all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
        # num_points = 0
        # for stroke in all_strokes:
        #     num_points += len(stroke)
        # avg_len = num_points / len(all_strokes)
        # tf.compat.v1.logging.info('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        #     len(all_strokes), len(train_strokes), len(valid_strokes),
        #     len(test_strokes), int(avg_len)))

        # # calculate the max strokes we need.
        # max_seq_len = utils.get_max_len(all_strokes)
        # # overwrite the hps with this calculation.
        # model_params.max_seq_len = max_seq_len

        # tf.compat.v1.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

        # eval_model_params = sketch_rnn_model.copy_hparams(model_params)

        # eval_model_params.use_input_dropout = 0
        # eval_model_params.use_recurrent_dropout = 0
        # eval_model_params.use_output_dropout = 0
        # eval_model_params.is_training = 1

        # if inference_mode:
        #     eval_model_params.batch_size = 1
        #     eval_model_params.is_training = 0

        # sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
        # sample_model_params.batch_size = 1  # only sample one at a time
        # sample_model_params.max_seq_len = 1  # sample one point at a time

        # train_set = utils.DataLoader(
        #     train_strokes,
        #     model_params.batch_size,
        #     max_seq_length=model_params.max_seq_len,
        #     random_scale_factor=model_params.random_scale_factor,
        #     augment_stroke_prob=model_params.augment_stroke_prob)

        # normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
        # train_set.normalize(normalizing_scale_factor)

        # valid_set = utils.DataLoader(
        #     valid_strokes,
        #     eval_model_params.batch_size,
        #     max_seq_length=eval_model_params.max_seq_len,
        #     random_scale_factor=0.0,
        #     augment_stroke_prob=0.0)
        # valid_set.normalize(normalizing_scale_factor)

        # test_set = utils.DataLoader(
        #     test_strokes,
        #     eval_model_params.batch_size,
        #     max_seq_length=eval_model_params.max_seq_len,
        #     random_scale_factor=0.0,
        #     augment_stroke_prob=0.0)
        # test_set.normalize(normalizing_scale_factor)

        # tf.compat.v1.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)

        # result = [
        #     train_set, valid_set, test_set, model_params, eval_model_params,
        #     sample_model_params
        # ]
        # return result
    
    # [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir, model_dir)
    # sketch_rnn_train.reset_graph()
    # myModel = model.Model(hps_model)
    # eval_model = model.Model(eval_hps_model, reuse=True)
    # sample_model = model.Model(sample_hps_model, reuse=True)
    # sess = tf.compat.v1.InteractiveSession()
    # sess.run(tf.compat.v1.global_variables_initializer())
    # sketch_rnn_train.load_checkpoint(sess, model_dir)

    # def encode(input_strokes):
    #     strokes = utils.to_big_strokes(input_strokes).tolist()
    #     strokes.insert(0, [0, 0, 1, 0, 0])
    #     seq_len = [len(input_strokes)]
    #     return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

    # def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
    #     z = None
    #     if z_input is not None:
    #         z = [z_input]
    #     sample_strokes, m = model.sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
    #     strokes = utils.to_normal_strokes(sample_strokes)
    #     return strokes

    # stroke = test_set.random_sample()
    # z = encode(stroke)
    # print(z)