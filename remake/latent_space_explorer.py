import tkinter as tk
import tensorflow as tf
import os
import svgwrite
import numpy as np
from IPython.display import SVG, display
import rdp

import utils

class ScalingCanvas(tk.Canvas):
    def __init__(self, sketch_window, parent, **kwargs):
        super(ScalingCanvas, self).__init__(parent, **kwargs)
        self.bind('<Configure>', self.__rescale_components)
        self.__height = self.winfo_reqheight()
        self.__width = self.winfo_reqwidth()
        self.__sketch_window = sketch_window

    def __rescale_components(self, event):
        # Find factors by which the canvas was resized
        width_factor = float(event.width)/self.__width
        height_factor = float(event.height)/self.__height

        # We also need do rescale the stroke magnitudes.
        # E.g. a 5 cm stroke will span a whole small canvas, but a small amount of a large one. But the stroke encoding should be the same.
        self.__sketch_window.adjustStrokeFactors((width_factor, height_factor))

        # Update width and height to new values
        self.__width = event.width
        self.__height = event.height

        # Actually resize the canvas
        self.config(width=self.__width, height=self.__height)

        # Re-scale all the components inside the canvas, args->(tags, offsetX, offsetY, xScale, yScale)
        self.scale('all', 0, 0, width_factor, height_factor)


class Sketch_Window:
    def __init__(self):
        # Erroneous Attributes
        self.__brushThickness = 10
        # Try to match the sample rate of 
        self.__sample_every = 2
        self.__sample_count = 1
        self.__cursor = (0, 0)
        self.__sketch = []
        self.__base_window_dims = (700, 600)
        self.__x_factor = 1
        self.__y_factor = 1

        # Main Window Propeties
        self.__root = tk.Tk()
        self.__root.title('Sketch Pad')
        self.__root.geometry('700x600')
        self.__root.minsize(400, 350)

        # Canvas to draw on
        self.__sketchCanvas = ScalingCanvas(self, self.__root, width=50, height=50)
        self.__sketchCanvas.config(bg='white', highlightbackground='black')
        self.__sketchCanvas.grid(row=0, column=0, columnspan=3, sticky='nesw')
        self.__sketchCanvas.bind('<B1-Motion>', self.__pen_down)
        self.__sketchCanvas.bind('<ButtonRelease-1>', self.__pen_up)

        # Buttons to initiate sketch generation or completion
        self.__generateButton = tk.Button(self.__root, text='Generate Similar Sketches', command=self.test)
        self.__finishButton = tk.Button(self.__root, text='Generate Finished Sketches')
        self.__generateButton.grid(row=1, column=0, columnspan=2, pady=(0, 5))
        self.__finishButton.grid(row=1, column=2, columnspan=2, pady=(0, 5))

        # Tools area
        self.__placeholder = tk.Canvas(self.__root, width=100, height=50)
        self.__placeholder.config(bg='blue', highlightbackground='black')
        self.__placeholder.grid(row=0, column=3, columnspan=1, sticky='nesw')

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
        self.__paint_oval(event)
        
        # Add movement to stroke list
        movementx = (event.x-self.__cursor[0])/self.__x_factor
        movementy = (event.y-self.__cursor[1])/self.__y_factor
        self.__sketch.append([movementx, movementy, 0])

        self.__cursor = (event.x, event.y)

    def __paint_oval(self, event):
        self.__sketchCanvas.create_oval(event.x-self.__brushThickness, event.y-self.__brushThickness, event.x+self.__brushThickness, event.y+self.__brushThickness, fill='black', outline='black')

    def __pen_up(self, event):
        self.__sketch[-1][-1] = 1

    def test(self):
        print(self.__sketch)
        print('£££')
        sketch = self.__selective_rdp(self.__sketch)
        print('$$$')
        print(sketch)
        draw_strokes(np.array(self.__sketch))

    def __selective_rdp(self, sketch, epsilon=0.2):
        # Use rdp to simplify a drawing, but specifically protect sections which correspond to 'pen up movements'.
        new_sketch = [sketch[0]]
        last_index = 1
        for i, stroke in enumerate(sketch[1:]):
            if stroke[2]:
                simplified_section = rdp.rdp(sketch[last_index:i+1][:-1], epsilon=epsilon)
                print(simplified_section[-1])
                for simplified_stroke in simplified_section:
                    new_sketch.append(simplified_stroke)
                new_sketch[-1][-1] = 1
                last_index = i+2
        simplified_section = rdp.rdp(sketch[last_index:][:-1], epsilon=epsilon)
        for simplified_stroke in simplified_section:
            new_sketch.append(simplified_stroke)
        return new_sketch

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = 'picture'):
  tf.compat.v1.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

if __name__ == '__main__':
    sketch_window = Sketch_Window()