# import tensorflow as tf
#
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow as tf

print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])
