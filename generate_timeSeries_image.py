from pyts import datasets
from T2I import generate_transform
import matplotlib.pyplot as plt


def generate_dataset_pic(method,dataset_name):
    gasf,gadf,RcP,Mark = generate_transform()
    #dataset_name = "Coffee"
    (data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset(dataset_name,return_X_y=True)
    data_size = data_train.shape[0]
    print(data_size)
    if data_size > 100:
        data_size = 100
    if method  == "Gasf":
        images_trian = gasf.fit_transform(data_train)
    elif method  == "Gadf":
        images_trian = gadf.fit_transform(data_train)
    elif method =="RcP":
        images_trian = RcP.fit_transform(data_train)
    elif method =="Mark":
        images_trian = Mark.fit_transform(data_train)
    print(images_trian.shape)

    fig, ax = plt.subplots(5, int(data_size/5-1), figsize=(96, 96))
    plt.rcParams['font.size'] = 100  # 设置字体的大小为10
    for i, axi in enumerate(ax.flat):
        axi.imshow(images_trian[i])
        axi.set(xticks=[], yticks=[],xlabel=target_train[i])

    fig.savefig('./images/{}train_{}.png'.format(dataset_name,method))

if __name__ == "__main__":
    dataset_list = datasets.ucr_dataset_list()
    print(dataset_list)
    #'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME', 'Beef'
    dataset_name = "Coffee"
    method = ["Gasf","Gadf","RcP","Mark"]
    generate_dataset_pic(method[2],dataset_name)


