import numpy as np
from sklearn import datasets

"""
8x8 tiny NIST image handler
NIST dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
If you chose NistHandle(flat=True), you'll get 64-dim vector instead of 8x8 numpy array.
Usage:
handler = NistHandle()
image = handler.get(1)  # you'll get 8x8 image of "one".
"""


class NistHandle:
    def __init__(self, flat=False):
        digits = datasets.load_digits()
        self.digit_dict = {}
        for n, target in enumerate(digits.target):
            image = digits.images[n] if not flat else digits.data[n]
            if self.digit_dict.get(target) is None:
                self.digit_dict[target] = [image]
            else:
                self.digit_dict[target].append(image)

    def get(self, target: int):
        digit_list = self.digit_dict[target]
        image_index = np.random.choice(len(digit_list))
        return digit_list[image_index]


if __name__ == '__main__':
    nh = NistHandle(flat=True)

    print(nh.get(0))
    print([len(nh.digit_dict[i]) for i in range(10)])