# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py


import json
import os
import cv2
import pathlib
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes


class VitonHDDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path,
            phase,
            tokenizer,
            order,
            radius = 5,
            caption_name = "captions.json",
            sketch_threshold_range = (20, 127),
            size = (512, 384),
            texture_patch_size_in_background = (350, 350),
            uncond_prob = 0.2,

    ):

        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_name = caption_name
        self.sketch_threshold_range = sketch_threshold_range
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.tokenizer = tokenizer
        self.order = order
        self.texture_patch_size = texture_patch_size_in_background
        self.uncond_prob = uncond_prob
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        im_names = []
        c_names = []
        dataroot_names = []

        # Load caption
        with open(os.path.join(self.dataroot, self.caption_name)) as f:
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items()}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        # 1. Read text description
        caption = self.captions_dict[c_name.split('_')[0]]
        if self.caption_name == "captions.json":
            caption = ", ".join(caption)

        original_caption = caption
        cond_input = self.tokenizer([caption], max_length=self.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
        max_length = cond_input.shape[-1]
        uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").input_ids
        caption = cond_input
        caption_uncond = uncond_input


        # 2. Read model image
        image = Image.open(os.path.join(dataroot, self.phase, 'image', im_name))
        image = image.resize((self.width, self.height))
        image = self.transform(image)  # [-1,1]
        
        # 3. Read Densepose
        densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name))
        densepose = densepose.resize((self.width, self.height))
        densepose = self.transform(densepose)  # [-1,1]

        # 4. Read sketch
        if self.order == 'unpaired':
            im_sketch = Image.open(
                os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png")))
        elif self.order == 'paired':
            im_sketch = Image.open(os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png")))
        else:
            raise ValueError(
                f"Order should be either paired or unpaired"
            )
        im_sketch = im_sketch.resize((self.width, self.height))
        im_sketch = ImageOps.invert(im_sketch)
        # threshold grayscale pil image
        im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
        im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
        im_sketch = 1 - im_sketch
        im_sketch = im_sketch.expand(3,-1,-1)

        # 5. Read texture image and paste it on white background
        texture = Image.open(os.path.join(self.dataroot, self.phase, 'cloth-texture', c_name))
        texture = texture.resize(self.texture_patch_size)
        # Create a white background
        background = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        # Calculate coordinates to paste texture_image to the center of background
        left = (background.width - texture.width) // 2
        top = (background.height - texture.height) // 2
        # Paste texture_image to the center of background
        background.paste(texture, (left, top))
        texture = self.transform(background)
        # --------------------------------------------------------------

        # Label Map
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(os.path.join(dataroot, self.phase, 'image-parse-v3', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        im_parse_final = transforms.ToTensor()(im_parse) * 255
        parse_array = np.array(im_parse)

        parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32)

        parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                            (parse_array == 2).astype(np.float32) + \
                            (parse_array == 18).astype(np.float32) + \
                            (parse_array == 19).astype(np.float32)

        parser_mask_changeable = (parse_array == 0).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)
        parse_mask = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)

        parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32)  # the lower body is fixed

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)
        # dilation
        parse_mask = parse_mask.cpu().numpy()
        # Load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
            pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

        pose_mapping = {
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:9,
            9:10,
            10:11,
            11:12,
            12:13,
            13:14,
            14:15,
            15:16,
            16:17,
            17:18
        }


        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)

        # do in any case because i have only upperbody
        with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
            data = json.load(f)
            data = data['people'][0]['pose_keypoints_2d']
            data = np.array(data)
            data = data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            data[:, 0] = data[:, 0] * (self.width / 768)
            data[:, 1] = data[:, 1] * (self.height / 1024)

            shoulder_right = np.multiply(tuple(data[pose_mapping[2]]), 1)
            shoulder_left = np.multiply(tuple(data[pose_mapping[5]]), 1)
            elbow_right = np.multiply(tuple(data[pose_mapping[3]]), 1)
            elbow_left = np.multiply(tuple(data[pose_mapping[6]]), 1)
            wrist_right = np.multiply(tuple(data[pose_mapping[4]]), 1)
            wrist_left = np.multiply(tuple(data[pose_mapping[7]]), 1)

            ARM_LINE_WIDTH = int(90 / 512 * self.height)
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                np.logical_not(
                                                                    np.array(parse_head_2, dtype=np.uint16))))

        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        # im_mask = image * parse_mask_total
        inpaint_mask = 1 - parse_mask_total

        # here we have to modify the mask and get the bounding box
        bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
        bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
        xmin = bboxes[0, 0]
        xmax = bboxes[0, 2]
        ymin = bboxes[0, 1]
        ymax = bboxes[0, 3]

        inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
            torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
            torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

        inpaint_mask = inpaint_mask.unsqueeze(0)
        im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))

        # ==========================Unconditional Training==========================
        # Generate 4 random floats between 0 and 1, representing probability order [Text,Pose,Sketch,Texture]
        if self.phase == "train":
            random_floats = np.random.rand(4) 
            for i in range(4):
                if random_floats[i] < self.uncond_prob:
                    if i == 0:
                        caption = caption_uncond
                        original_caption = ""
                    elif i == 1:
                        densepose = torch.zeros_like(densepose)
                    elif i == 2:
                        im_sketch = torch.zeros_like(im_sketch)
                    elif i == 3:
                        texture = torch.zeros_like(texture)

        result = {
            "c_name": c_name,
            "im_name": im_name,
            "image": image,
            "texture": texture,
            "caption": caption,
            "original_caption": original_caption,
            "im_sketch": im_sketch,
            "densepose": densepose,
            "im_mask": im_mask,
            "inpaint_mask": inpaint_mask,
            "im_parse": im_parse_final
        }

        return result

    def __len__(self):
        return len(self.c_names)


label_map={
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

class DressCodeDataset(data.Dataset):
    def __init__(self,
                 dataroot_path,
                 phase,
                 tokenizer,
                 sketch_threshold_range,
                 order,
                 size,
                 uncond_prob = 0.2,
                 radius=5,
                 category=['dresses', 'upper_body', 'lower_body'],
                 caption_folder='fine_captions.json',
                 coarse_caption_folder='coarse_captions.json',
                 texture_patch_size_in_background = (350, 350),
                 ):

        super(DressCodeDataset, self).__init__()
        self.dataroot = pathlib.Path(dataroot_path)
        self.phase = phase
        self.caption_folder = caption_folder
        self.sketch_threshold_range = sketch_threshold_range
        self.category = category
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.uncond_prob = uncond_prob
        self.texture_patch_size = texture_patch_size_in_background
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        allowed_combinations = {
            ("train", "paired"),
            ("test", "unpaired"),
            ("test", "paired")
        }

        assert (phase, order) in allowed_combinations, "Invalid phase-order combination"
        im_names = []
        c_names = []
        dataroot_names = []


        # Load caption
        with open(self.dataroot / self.caption_folder) as f:
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        with open(self.dataroot / coarse_caption_folder) as f:
            self.captions_dict.update(json.load(f))

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = self.dataroot / c
            if phase == 'train':
                filename = dataroot / f"{phase}_pairs.txt"
            else:
                filename = dataroot / f"{phase}_pairs_{order}.txt"

            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    if c_name.split('_')[0] not in self.captions_dict:
                        continue

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """

        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = str(dataroot.name)
        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        # 1. Read captions
        caption = self.captions_dict[c_name.split('_')[0]]
        # if train randomly shuffle caption if there are multiple, else concatenate with comma
        if self.phase == 'train':
            random.shuffle(caption)
        caption = ", ".join(caption)

        original_caption = caption

        cond_input = self.tokenizer([caption], max_length=self.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
        max_length = cond_input.shape[-1]
        uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").input_ids
        caption = cond_input
        caption_uncond = uncond_input
        
        # 2. Read model image
        image = Image.open(dataroot / 'images' / im_name)

        image = image.resize((self.width, self.height))
        image = self.transform(image)  # [-1,1]
        # 3. Read densepose
        densepose = Image.open(dataroot / 'dense' / im_name.replace('0.jpg', '5.png')).convert('RGB')
        densepose = densepose.resize((self.width, self.height))
        densepose = self.transform2D(densepose)
        # 4. Read texture image and paste it on white background
        texture = Image.open(dataroot / f'{category}_cloth-texture'/ c_name)
        texture = texture.resize(self.texture_patch_size)
        # Create a white background
        background = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        # Calculate coordinates to paste texture_image to the center of background
        left = (background.width - texture.width) // 2
        top = (background.height - texture.height) // 2
        # Paste texture_image to the center of background
        background.paste(texture, (left, top))
        texture = self.transform(background)
        # 5. Read im_sketch
        if "unpaired" == self.order and self.phase == 'test':  # Upper of multigarment is the same of unpaired
            im_sketch = Image.open(
                dataroot / 'im_sketch_unpaired' / f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}')
        else:
            im_sketch = Image.open(dataroot / 'im_sketch' / c_name.replace(".jpg", ".png"))

        im_sketch = im_sketch.resize((self.width, self.height))
        im_sketch = ImageOps.invert(im_sketch)
        # threshold grayscale pil image
        im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
        # im_sketch = im_sketch.convert("RGB")
        im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
        im_sketch = 1 - im_sketch
        im_sketch = im_sketch.expand(3,-1,-1)
        # 6. Load head
        if self.phase == 'test':
            stitch_labelmap = Image.open(self.dataroot / 'test_stitch_map' / im_name.replace(".jpg", ".png"))
            stitch_labelmap = transforms.ToTensor()(stitch_labelmap) * 255
            stitch_label = stitch_labelmap == 13
        # 7. Load parse, used to generate im_mask and inpaint_mask
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(dataroot / 'label_maps' / parse_name)
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        parse_array = np.array(im_parse)


        parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 3).astype(np.float32) + \
                        (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        
        if category == 'dresses':
            parse_cloth = (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 7).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32) + \
                            (parse_array == 13).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif category == 'upper_body':
            parse_cloth = (parse_array == 4).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                    (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif category == 'lower_body':
            parse_cloth = (parse_array == 6).astype(np.float32)
            parse_mask = (parse_array == 6).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32) + \
                            (parse_array == 13).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                    (parse_array == 14).astype(np.float32) + \
                                    (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        else:
            raise NotImplementedError

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_mask = parse_mask.cpu().numpy()
        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')

        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)
        if category == 'dresses' or category == 'upper_body' or category == 'lower_body':
            with open(dataroot / 'keypoints' / pose_name, 'r') as f:
                data = json.load(f)
                shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', 45, 'curve')

            hands = np.logical_and(np.logical_not(im_arms), arms)

            if category == 'dresses' or category == 'upper_body':
                parse_mask += im_arms
                parser_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)
        if category == 'dresses' or category == 'upper_body':
            with open(dataroot / 'keypoints' / pose_name, 'r') as f:
                data = json.load(f)
                points = []
                points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                x_coords, y_coords = zip(*points)
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                for i in range(parse_array.shape[1]):
                    y = i * m + c
                    parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                np.logical_not(
                                                                    np.array(parse_head_2, dtype=np.uint16))))

        # tune the amount of dilation here
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        im_mask = image * parse_mask_total
        inpaint_mask = 1 - parse_mask_total

        # here we have to modify the mask and get the bounding box
        bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
        bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
        xmin = bboxes[0, 0]
        xmax = bboxes[0, 2]
        ymin = bboxes[0, 1]
        ymax = bboxes[0, 3]

        inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
            torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
            torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

        inpaint_mask = inpaint_mask.unsqueeze(0)
        im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
        # ==========================Unconditional Training==========================
        # Generate 4 random floats between 0 and 1, representing probability order [Text,Pose,Sketch,Texture]
        if self.phase == "train":
            random_floats = np.random.rand(4) 
            for i in range(4):
                if random_floats[i] < self.uncond_prob:
                    if i == 0:
                        caption = caption_uncond
                        original_caption = ""
                    elif i == 1:
                        densepose = torch.zeros_like(densepose)
                    elif i == 2:
                        im_sketch = torch.zeros_like(im_sketch)
                    elif i == 3:
                        texture = torch.zeros_like(texture)
        
        
        result = {
            "category": category,
            "c_name": c_name,
            "im_name": im_name,
            "image": image,
            "densepose": densepose,
            "caption": caption,
            "texture": texture,
            "original_caption": original_caption,
            "im_sketch": im_sketch,
            "im_mask": im_mask,
            "inpaint_mask": inpaint_mask
        }
        if self.phase == "test":
            result["stitch_label"] = stitch_label

        return result

    def __len__(self):
        return len(self.c_names)
