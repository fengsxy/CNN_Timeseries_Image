from pyts import datasets
from T2I import generate_transform
import matplotlib.pyplot as plt
gasf,gadf,RcP,Mark = generate_transform()
(data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("Coffee",return_X_y=True)

gasf_images_trian = gasf.fit_transform(data_train)
#gadf_images_trian = gadf.fit_transform(data_train)
#RcP_images_trian = RcP.fit_transform(data_train)
#Mark_images_trian = Mark.fit_transform(data_train)
print(gasf_images_trian.shape)
fig, ax = plt.subplots(4, 7, figsize=(96, 96))
for i, axi in enumerate(ax.flat):
    axi.imshow(gasf_images_trian[i])
    axi.set(xticks=[], yticks=[])
plt.show()