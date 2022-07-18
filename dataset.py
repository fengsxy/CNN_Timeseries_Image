import torch.utils.data as data


class Mydataset(data.Dataset):

    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]

    def __len__(self):
        return len(self.x_list)

