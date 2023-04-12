import tensorflow as tf

import svgwrite
import os
import rdp


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

        min_x, max_x, min_y, max_y = DataUtilities.get_bounds(
            sketch,
            size_coefficient
        )
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
