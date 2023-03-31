from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

import numpy as np
import tensorflow as tf
import svgwrite, os, utils
import requests
from io import BytesIO

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

def draw_strokes(data, svg_filename, factor=0.2):
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
    x = float(data[i, 0])/factor
    y = float(data[i, 1])/factor
    colour_coeff = data[i, 2]
    lift_pen = data[i, 3]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()

def new_draw_strokes(data, filename, factor=0.2):
  tf.compat.v1.gfile.MakeDirs(os.path.dirname(filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  x = abs_x
  y = abs_y
  lift_pen = 1
  colour_coeff = 1
  for i in range(len(data)):
    x += float(data[i][0])/factor
    y += float(data[i][1])/factor
    if lift_pen:
      if not data[i][3]:
        p = f'M{x},{y} '
        colour_coeff = data[i][2]
    else:
      p += f'L{x},{y} '
      if data[i][3]:
        dwg.add(dwg.path(p).stroke(f'rgb({int(255*(1-colour_coeff))}, {int(255*(1-colour_coeff))}, {int(255*(1-colour_coeff))})', 1).fill('none'))
    lift_pen = data[i][3]
  dwg.save()


def load_data(filepath, display=True):
  if filepath.startswith('http') or filepath.startswith('https'):
    response = requests.get(filepath, allow_redirects=True)
    data = np.load(BytesIO(response.content), allow_pickle=True, encoding='latin1')
  else:
    data = np.load(filepath, allow_pickle=True)

  if display:
    print(f"Loaded {len(data['train'])}/{len(data['test'])}/{len(data['valid'])} (train, test, valid) sketches from {filepath}")

  return data['train'], data['test'], data['valid']

def create_svgs(data):
    for i, image in enumerate(data):
        if i%100==0:
            print(i)
        draw_strokes(image, 'data/augmented/svgs/'+str(i))

def convert_to_png(svg_filepath, destination_filepath):
    drawing = svg2rlg(svg_filepath)
    renderPM.drawToFile(drawing, destination_filepath, fmt="PNG")

def convert_svgs_to_pngs(svg_filepath, png_filepath):
    for i in range(8000):
        if i%100==0:
            print(i)
        drawing = svg2rlg(svg_filepath+str(i))
        renderPM.drawToFile(drawing, png_filepath+str(i)+'.png')

def convert_to_4_stroke(data):
  # convert stroke-3 (npz) to stroke-4
  new_data = []
  for movement in data:
    conv_movement = movement.tolist()
    new_movement = conv_movement[:2] + [1] + conv_movement[2:]
    new_data.append(new_movement)
  return new_data

def augment_exponential_colour_decay(data, decay_rate=0.98):
  # Expects stroke-4 format
  new_data = []
  colour = 1
  for i, movement in enumerate(data):
    new_data.append(movement[:2] + [colour] + movement[3:])
    if movement[3]==1:
      colour = decay_rate**i
  return new_data

def convert_inners_to_np(dataset):
  for i in range(len(dataset)):
    dataset[i] = np.array(dataset[i], dtype=np.float32)

def augment_dataset(dataset, func):
  new_dataset = []
  for item in dataset:
    new_dataset.append(func(item))
  return new_dataset

if __name__ == '__main__':

    # train, test, valid = load_data('https://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep/aaron_sheep.npz')

    # new_train = augment_dataset(train, convert_to_4_stroke)
    # new_train = augment_dataset(new_train, augment_exponential_colour_decay)
    # convert_inners_to_np(new_train)

    # new_test = augment_dataset(test, convert_to_4_stroke)
    # new_test = augment_dataset(new_test, augment_exponential_colour_decay)
    # convert_inners_to_np(new_test)

    # new_valid = augment_dataset(valid, convert_to_4_stroke)
    # new_valid = augment_dataset(new_valid, augment_exponential_colour_decay)
    # convert_inners_to_np(new_valid)

    # np.savez_compressed('data/stroke_4/sheep_colour_decay', train=new_train, test=new_test, valid=new_valid)

    train, test, valid = load_data('data/stroke_4/sheep_colour_decay.npz')