import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import xarray

class dataFetcher(Dataset):

    def __init__(self, data_path):
        #loading the dataset
        dataset1 = xarray.open_dataset(data_path).load()
        #normalizing the wave length and passing the centre and span
        self.transmittance = torch.from_numpy(dataset1.efficiency.values)
        #taking the numpy array of patterns to torch
        self.patterns = torch.from_numpy(dataset1.pattern.values).unsqueeze(1)
        #transforming (resizing and enabling the access to GPU)
        self.transfrom = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((32,128)),
                                transforms.ToTensor()])

    def normalizeData(self, tensorData):
        maxValue = torch.max(tensorData)
        minValue = torch.min(tensorData)
        # calculating average
        average = (maxValue + minValue)/ 2.0
        #calculating the range
        range = (maxValue - minValue)/ 2.0
        #noramlize each value in the tensor array
        normTensorData = (tensorData - average) / range
        return normTensorData, average, range

    def __len__(self):
        #getting the length of the dataset
        return len(self.transmittance)

    def __getitem__(self, idx):
        #getting the transmittance in the label of float values
        label1 = torch.FloatTensor([self.transmittance[idx]])
        # img = random_x_translation(self.patterns[idx])
        #translating
        translation = int(torch.randint(0, self.patterns[idx].size(-1), (1,)))
        img = torch.cat([self.patterns[idx][..., translation:], self.patterns[idx][..., :translation]], dim=-1)
        img = (self.transfrom(img) - 0.5) / 0.5
        return img, label1
