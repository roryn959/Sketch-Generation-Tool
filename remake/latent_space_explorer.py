import tkinter as tk

class ScalingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super(ScalingCanvas, self).__init__(parent, **kwargs)
        self.bind('<Configure>', self.__rescale_components)
        self.__height = self.winfo_reqheight()
        self.__width = self.winfo_reqwidth()

    def __rescale_components(self, event):
        # Find factors by which the canvas was resized
        width_factor = float(event.width)/self.__width
        height_factor = float(event.height)/self.__height

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
        self.__cursor = (0, 0)
        self.__currentSketch = []

        # Main Window Propeties
        self.__root = tk.Tk()
        self.__root.title('Sketch Pad')
        self.__root.geometry('700x600')
        self.__root.minsize(400, 350)

        # Canvas to draw on
        self.__sketchCanvas = ScalingCanvas(self.__root, width=50, height=50)
        self.__sketchCanvas.config(bg='white', highlightbackground='black')
        self.__sketchCanvas.grid(row=0, column=0, columnspan=3, sticky='nesw')
        self.__sketchCanvas.bind('<B1-Motion>', self.__pen_down)
        self.__sketchCanvas.bind('<ButtonRelease-1>', self.__pen_up)

        # Buttons to initiate sketch generation or completion
        self.__generateButton = tk.Button(self.__root, text='Generate Similar Sketches')
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

    def __pen_down(self, event):
        self.__paint_oval(event)
        
        # Add movement to stroke list
        

    def __paint_oval(self, event):
        self.__sketchCanvas.create_oval(event.x-self.__brushThickness, event.y-self.__brushThickness, event.x+self.__brushThickness, event.y+self.__brushThickness, fill='black', outline='black')

    def __pen_up(self, event):
        print('pen up')

if __name__ == '__main__':
    sketch_window = Sketch_Window()