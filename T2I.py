from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField


def generate_transform():
    img_sz = 96  # 确定生成的 GAF 图片的大小
    method_1 = 'summation'  # GAF 图片的类型，可选 'summation'（默认）和 'difference'
    gasf = GramianAngularField(image_size=img_sz, method=method_1)
    method_2 = 'difference'  # GAF 图片的类型，可选 'summation'（默认）和 'difference'
    gadf = GramianAngularField(image_size=img_sz, method=method_2)
    RcP = RecurrencePlot()
    Mark = MarkovTransitionField()
    return  gasf,gadf,RcP,Mark
