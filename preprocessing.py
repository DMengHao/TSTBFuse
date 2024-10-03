import os
import pickle
from skimage.io import imread
import numpy as np
from tqdm import tqdm
train = True
file = './data'
if not os.path.isdir(file):
    os.makedirs(file)
if train == True:
    ir_path = './Datasets/train/MSRS/ir'
    vi_path = './Datasets/train/MSRS/vi'
    pkl = './data/train_datasets.pkl'
else:
    ir_path = './Datasets/val/MSRS/ir'
    vi_path = './Datasets/val/MSRS/vi'
    pkl = './data/val_datasets.pkl'
data = {}
def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    # 亮度或强度对应的比例
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
class preprocessing():
    def __init__(self):
        super(preprocessing, self).__init__()
        pass
    @classmethod
    def get_dataset_files(cls, files_path):
        image_list = []
        for root, dirs, files in os.walk(files_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')) != True:
                    continue
                else:
                    image_list.append(os.path.join(root, file))
        return image_list

    @classmethod
    def RGB_to_2Y(cls, image):
        # image size is (C, H, W)
        Y = image[0:1, :, :] * 0.299000 + image[1:2, :, :] * 0.587000 + image[2:3, :, :] * 0.114000
        return Y

    @classmethod
    def get_data(cls, train: bool):
        data = {}
        ir = []
        vi = []
        ir_list = sorted(preprocessing.get_dataset_files(ir_path))
        vi_list = sorted(preprocessing.get_dataset_files(vi_path))
        assert len(ir_list) == len(vi_list), print('ir images and vi images not equal')
        print("######## Start processing infrared and visible light images... ########")
        for i in tqdm(range(len(ir_list))):
            ir_image = imread(ir_list[i]).astype(np.float32)[None, :, :]/255.0
            I_IR_Patch_Group = Im2Patch(ir_image, 128, 200)
            # ir.append(ir_image)
            vi_image = np.transpose(imread(vi_list[i]).astype(np.float32),axes=(2,0,1))/255.0
            vi_image = preprocessing.RGB_to_2Y(vi_image)
            I_VIS_Patch_Group = Im2Patch(vi_image, 128, 200)
            for ii in range(I_IR_Patch_Group.shape[-1]):
                bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii])
                bad_VIS = is_low_contrast(I_VIS_Patch_Group[0, :, :, ii])
                # Determine if the contrast is low
                if not (bad_IR or bad_VIS):
                    avl_IR = I_IR_Patch_Group[0, :, :, ii]  # available IR
                    avl_VIS = I_VIS_Patch_Group[0, :, :, ii]
                    avl_IR = avl_IR[None, ...]
                    avl_VIS = avl_VIS[None, ...]
                    ir.append(avl_IR)
                    vi.append(avl_VIS)
        data['ir'] = ir
        data['vi'] = vi
        if train == True:
            with open('./data/train_datasets.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open('./data/val_datasets.pkl', 'wb') as f:
                pickle.dump(data, f)

if __name__ == '__main__':
    print("##################### Begin Preprocessing! ########################")
    preprocessing.get_data(train)
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    print("##################### Preprocessing Done! ########################")
