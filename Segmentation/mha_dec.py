import os
import numpy as np
from tqdm import tqdm
from medpy.io import load
import skimage.io as io
import skimage.transform as trans


mha_directory = 'path to HGG or LGG directory'
npy_dir = 'path to npy directory where converted files should be saved'

#Counters for all image types
T1_counter=0
T2_counter=0
T1c_counter=0
flair_counter=0
OT_counter=0


def read_resample_resize_mha(path):
  global T1_counter
  global T2_counter
  global T1c_counter
  global flair_counter
  global OT_counter


  for (directory, subdirectories, files) in os.walk(path):

    for image in files:

      if '.mha' in image:
        image_data, image_header = load(directory+'/'+image)
        print(image_data.shape)
        if image_data.shape[0] is not 240:
          print('height is not 240 but ' + str(image_data.shape[0]))
          raise RuntimeError("Wrong height")
        elif image_data.shape[1] is not 240:
          print('width is not 240 but ' + str(image_data.shape[1]))
          raise RuntimeError("Wrong width")
        elif image_data.shape[2] is not 155:
          print('depth is not 155 but ' + str(image_data.shape[2]))
          raise RuntimeError("Wrong depth")

        if 'T1c' in image:
          print(image, 'IMAGE T1c')
          if not os.path.exists(npy_dir + '/T1c'):
            os.makedirs(npy_dir + '/T1c')

          img = io.imread(directory+'/'+image, plugin='simpleitk')
          img = (img - img.mean()) / img.std()
          img = trans.resize(img, (128,128,128), mode='constant')

          np.save(os.path.join(npy_dir, 'T1c', f'{T1c_counter:04}.npy'), np.array(img)[..., np.newaxis].astype('float32'))
          T1c_counter = T1c_counter+1

        elif 'T1' in image:
          print(image, 'IMAGE T1')
          if not os.path.exists(npy_dir + '/T1'):
            os.makedirs(npy_dir + '/T1')

          img = io.imread(directory + '/' + image, plugin='simpleitk')
          img = (img - img.mean()) / img.std()
          img = trans.resize(img, (128, 128, 128), mode='constant')

          np.save(os.path.join(npy_dir, 'T1', f'{T1_counter:04}.npy'), np.array(img)[..., np.newaxis].astype('float32'))
          T1_counter = T1_counter + 1

        elif 'T2' in image:
          print(image, 'IMAGE T2')
          if not os.path.exists(npy_dir + '/T2'):
            os.makedirs(npy_dir + '/T2')

          img = io.imread(directory + '/' + image, plugin='simpleitk')
          img = (img - img.mean()) / img.std()
          img = trans.resize(img, (128, 128, 128), mode='constant')

          np.save(os.path.join(npy_dir, 'T2', f'{T2_counter:04}.npy'), np.array(img)[..., np.newaxis].astype('float32'))
          T2_counter = T2_counter + 1

        elif 'Flair' in image:
          print(image, 'IMAGE FLAIR')
          if not os.path.exists(npy_dir + '/Flair'):
            os.makedirs(npy_dir + '/Flair')

          img = io.imread(directory + '/' + image, plugin='simpleitk')
          img = (img - img.mean()) / img.std()
          img = trans.resize(img, (128, 128, 128), mode='constant')

          np.save(os.path.join(npy_dir, 'Flair', f'{flair_counter:04}.npy'), np.array(img)[..., np.newaxis].astype('float32'))
          flair_counter = flair_counter + 1

        elif 'OT' in image:
          print(image, 'IMAGE OT')
          if not os.path.exists(npy_dir + '/OT'):
            os.makedirs(npy_dir + '/OT')

          img = io.imread(directory + '/' + image, plugin='simpleitk')
          img[img == 4] = 1
          img[img != 1] = 0
          img = img.astype('float32')
          img = trans.resize(img, (128,128,128), mode='constant')

          np.save(os.path.join(npy_dir, 'OT', f'{OT_counter:04}.npy'), np.array(img)[..., np.newaxis].astype('float32'))

          OT_counter = OT_counter + 1

  return img, 'T1c'

def get_mha_paths(root):
  for (directory, subdirectories, files) in os.walk(root):
    if any(path.endswith('.mha') for path in os.listdir(directory)):
      yield directory

def get_mha_iterator(root):
  for path in get_mha_paths(root):
    try:
      array = read_resample_resize_mha(path)
    except RuntimeError as e:
      print("Continuing...")
      continue
    else:
      yield array

for i, (arrays) in enumerate(tqdm(get_mha_iterator(mha_directory))):

  for j, array in enumerate(arrays):

    npy_dir = os.path.join(npy_dir)
    if not os.path.exists(npy_dir):
      os.makedirs(npy_dir)







