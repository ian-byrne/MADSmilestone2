"""ML models and data definitions utilized within the Milestone2 project."""
import torch.nn as nn
import boto3
import torch
import torchvision
import torchvision.transforms as transforms
import tempfile
from PIL import Image
from skimage.io import imread
import botocore
import numpy as np
import torch.nn.functional as F


class ResizedClocks:
    """Resized clock drawing dataset."""

    def __init__(self, round, round_labels, pubkey, seckey, normalize_=None):
        """Define the dataset.

        Args:
            round (int): Round to grab images from.
            values (list of tuples): Corresponding values for the round.
        """
        self.round = round
        self.vals = round_labels
        self.client = boto3.client(
            "s3", aws_access_key_id=pubkey, aws_secret_access_key=seckey
        )
        if normalize_ == True:
            processes = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            rgb_val = True
        else:
            processes = transforms.ToTensor()
            rgb_val = None
        self.transform = processes
        self.rgb = rgb_val

    def __len__(self):
        """Define dataset length."""
        return len(self.vals)

    # def get_labels(self, idx):
    # return self.vals[idx][1]#self.vals[:, 1]

    def __getitem__(self, idx):
        """Loops through indexed items in dataset."""
        spid = self.vals[idx][0]
        label = torch.tensor(int(self.vals[idx][1]))
        bucket = "clockimages"  # "test-bucket-clockids-aicrowd"
        obj_name = f"NHATS_R{self.round}_ClockDrawings/{spid}.tif"  # f"{self.round}_{spid}.tif"
        # filename = str(spid)+".tif"
        temp = tempfile.NamedTemporaryFile()
        try:
            self.client.download_file(bucket, obj_name, temp.name)  # filename)

            im = Image.open(temp.name)  # filename)

            if self.rgb == True:
                # print('rgb')
                gray = im.convert("RGB")
                resized = gray.resize((284, 368))
                im_arr = np.array(resized)

            else:
                # print('gray')
                gray = im.convert("1")
                resized = gray.resize((284, 368))
                im_arr = np.float32(np.array(resized))

            # resized = gray.resize((284, 368))#(160, 207))##(2560, 3312))
            # resized = gray.resize((512, 662))
            # im_arr = np.float32(np.array(resized))#.astype(float)
            # im_arr = np.array(resized)

            if self.transform:
                im_arr = self.transform(im_arr)

            # sample = {'image': im_arr, 'label': label}

            temp.close()

            return im_arr, label

        except botocore.exceptions.ClientError as e:
            return

        # try:
        #     self.client.download_file(bucket, obj_name, temp.name)  # filename)

        #     im = Image.open(temp.name)  # filename)

        #     if self.rgb == True:
        #         # print('rgb')
        #         gray = im.convert("RGB")

        #     else:
        #         # print('gray')
        #         gray = im.convert("1")

        #     resized = gray.resize((284, 368))  # (160, 207))##(2560, 3312))
        #     # resized = gray.resize((512, 662))
        #     # im_arr = np.float32(np.array(resized))#.astype(float)
        #     im_arr = np.array(resized)

        #     if self.transform:
        #         im_arr = self.transform(im_arr)

        #     # sample = {'image': im_arr, 'label': label}

        #     temp.close()

        #     return im_arr, label

        # except botocore.exceptions.ClientError as e:
        #     return


# original size: 2560, 3312
class ConvNet(nn.Module):
    """From scratch CNN to label dementia."""

    def __init__(self):
        """Define CNN."""
        super(ConvNet, self).__init__()

        # without considering batch size: Input shape : (None,368, 284, 1) , parameters: (3*3*1*16+16) = 160
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,  # one input channel gray scale, 16 filters out
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Out:(None,386, 284, 16). ### TRY kernel 7x7 padding 3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.pool1 = nn.MaxPool2d(2, 2)  # Out: (None, 184, 142, 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool2 = nn.MaxPool2d(2, 2)  # Output shape = (None, 92, 71, 64)
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128,
        # kernel_size = 3, stride = 1, padding = 1) # params: (3*3*32*32+32) = 9248
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool3 = nn.MaxPool2d(2, 2)  # Output shape = (None, 46, 35, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout(0.3)

        # Fully connected layer
        self.fc1 = nn.Linear(
            128 * 46 * 35, 60
        )  # most recent original size of: 512, 662 -->64 x 82
        self.do3 = nn.Dropout(0.4)  # 40 % probability
        # self.fc3 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(60, 3)  # left with 3 for the three classes

    # def forward(self, x):
    #     """Feed through network."""
    #     x = self.bn1(
    #         self.pool1(F.relu(self.conv2(spectral_norm(F.relu(self.conv1(x))))))
    #     )
    #     x = self.bn2(
    #         self.pool2(F.relu(self.conv4(spectral_norm(F.relu(self.conv3(x))))))
    #     )
    #     # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
    #     x = self.bn3(self.pool3(spectral_norm(F.relu(self.conv6((x))))))
    #     x = self.do2(x)
    #     x = x.view(x.size(0), 128 * 64 * 82)
    #     x = spectral_norm(F.relu(self.fc1(x)))
    #     x = self.do3(x)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        """Feed through network."""
        x = self.bn1(self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.bn2(self.pool2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
        x = self.bn3(self.pool3(F.relu(self.conv6((x)))))
        x = self.do2(x)
        x = x.view(x.size(0), 128 * 46 * 35)
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)
        return x


class ConvNetScores(nn.Module):
    """From scratch CNN to score the clocks."""

    def __init__(self):
        """Define CNN."""
        super(ConvNet, self).__init__()

        # without considering batch size: Input shape : (None,368, 284, 1) , parameters: (3*3*1*16+16) = 160
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,  # one input channel gray scale, 16 filters out
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Out:(None,386, 284, 16). ### TRY kernel 7x7 padding 3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.pool1 = nn.MaxPool2d(2, 2)  # Out: (None, 184, 142, 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool2 = nn.MaxPool2d(2, 2)  # Output shape = (None, 92, 71, 64)
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128,
        # kernel_size = 3, stride = 1, padding = 1) # params: (3*3*32*32+32) = 9248
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool3 = nn.MaxPool2d(2, 2)  # Output shape = (None, 46, 35, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout(0.3)

        # Fully connected layer
        self.fc1 = nn.Linear(
            128 * 46 * 35, 60
        )  # most recent original size of: 512, 662 -->64 x 82
        self.do3 = nn.Dropout(0.4)  # 40 % probability
        # self.fc3 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(60, 6)  # left with 3 for the three classes

    # def forward(self, x):
    #     """Feed through network."""
    #     x = self.bn1(
    #         self.pool1(F.relu(self.conv2(spectral_norm(F.relu(self.conv1(x))))))
    #     )
    #     x = self.bn2(
    #         self.pool2(F.relu(self.conv4(spectral_norm(F.relu(self.conv3(x))))))
    #     )
    #     # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
    #     x = self.bn3(self.pool3(spectral_norm(F.relu(self.conv6((x))))))
    #     x = self.do2(x)
    #     x = x.view(x.size(0), 128 * 64 * 82)
    #     x = spectral_norm(F.relu(self.fc1(x)))
    #     x = self.do3(x)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        """Feed through network."""
        x = self.bn1(self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.bn2(self.pool2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
        x = self.bn3(self.pool3(F.relu(self.conv6((x)))))
        x = self.do2(x)
        x = x.view(x.size(0), 128 * 46 * 35)
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)
        return x
