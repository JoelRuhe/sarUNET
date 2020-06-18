import os
import shutil
import glob


def load_dataset(args, train):
  print(train, 'TRAIN VALUE')
  image_type = args.image_type
  # scratch_dir = '/home/joelr/scratch/'
  scratch_dir = args.scratch_path

  if train:
    npy_dir = args.dataset_root+'/train/'+args.image_tissue+'/'+image_type+'/'
    seg_dir = args.dataset_root+'/train/'+args.image_tissue+'/OT/'

  else:
    npy_dir = args.dataset_root+'/test/HGG_LGG/' + image_type + '/'

  MRIimages = []
  SEGimages = []

  if not os.path.exists(scratch_dir):
    print("Creating scratch directory")
    os.makedirs(scratch_dir)

  if train:
    npy_files = glob.glob(npy_dir + '*.npy')
    seg_files = glob.glob(seg_dir + '*.npy')
  else:
    npy_files = glob.glob(npy_dir + '*.npy')

  # Copying segmentation images
  if train:
    print('Copying segmentation (OT) images to scratch...')
    for file in seg_files:
      scratch_path = os.path.join(scratch_dir, args.image_tissue, 'OT')
      if not os.path.isdir(os.path.normpath(scratch_path)):
        os.makedirs(scratch_path)
      if not os.path.isfile(os.path.normpath(scratch_path + file)):
        shutil.copy(file, os.path.normpath(scratch_path))

  #Copying normal MRI brain images
  print('Copying '+image_type+' images to scratch...')
  for file in npy_files:
    scratch_path = os.path.join(scratch_dir, args.image_tissue, image_type)
    if not os.path.isdir(os.path.normpath(scratch_path)):
      os.makedirs(scratch_path)
    if not os.path.isfile(os.path.normpath(scratch_path + file)):
      shutil.copy(file, os.path.normpath(scratch_path))

  # Append segmenation images to array
  if train:
    scratch_path = os.path.join(scratch_dir, args.image_tissue, 'OT')
    for files in os.walk(scratch_path):
      for file in files[1:]:
        for f in file:
          if '.npy' in f:
            SEGimages.append(os.path.join(scratch_path, f))

  scratch_path = os.path.join(scratch_dir, args.image_tissue, image_type)
  for files in os.walk(scratch_path):
    for file in files[1:]:
      for f in file:
        if '.npy' in f:
         MRIimages.append(os.path.join(scratch_path, f))

  if train:
    return MRIimages, SEGimages
  else:
    return MRIimages
