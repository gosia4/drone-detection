import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from yolo_v3_model import yolo_v3
from dataset import Dataset
from utils import read_class_names, decode, loss_func

# Numerical checks function
def check_numerics(tensor, message):
    if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
        tf.print(f"{message} contains nan or inf values: {tensor}")
        raise ValueError(f"{message} contains nan or inf values")

""" 
Main function to train YOLO_V3 model
"""

# Train options
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_CLASSES = "thermographic_data/classes.txt"
TRAIN_NUM_OF_CLASSES = len(read_class_names(TRAIN_CLASSES))
TRAIN_MODEL_NAME = "model_2"
TRAIN_ANNOT_PATH = "thermographic_data/train"
TRAIN_LOGDIR = "log" + '/' + TRAIN_MODEL_NAME
TRAIN_CHECKPOINTS_FOLDER = "checkpoints" + '/' + TRAIN_MODEL_NAME
TRAIN_BATCH_SIZE = 2 # wcześniej było 4 (czyli wymagało 4 kolejne sekwencje klatek)
TRAIN_INPUT_SIZE = 416
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
# TRAIN_LR_INIT = 1e-5
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 1
TRAIN_EPOCHS = 30
TRAIN_DECAY = 0.8
TRAIN_DECAY_STEPS = 50.0

# TEST options
TEST_ANNOT_PATH = "thermographic_data/validate"
TEST_BATCH_SIZE = 2
TEST_INPUT_SIZE = 416
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45

# YOLO options
YOLO_STRIDES = [8, 16, 32]
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416
YOLO_BATCH_FRAMES = 2
YOLO_PREPROCESS_IOU_THRESH = 0.3
YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]


images_folder = r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/train/images/free_1"
labels_folder = r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/train/labels/free_1"


def load_bounding_box_points_from_label_file(label_file):
    points = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                # Znormalizowane wartości cx, cy, width, height
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tl = (cx - w / 2, cy - h / 2)
                tr = (cx + w / 2, cy - h / 2)
                bl = (cx - w / 2, cy + h / 2)
                br = (cx + w / 2, cy + h / 2)
                points.extend([tl, tr, bl, br])
    return points


def display_images_with_crosses(images_folder, labels_folder):
    for image_filename in os.listdir(images_folder):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_filename)

            image = cv2.imread(image_path)
            if image is None:
                continue

            # Wczytaj współrzędne punktów z pliku etykiet
            points = load_bounding_box_points_from_label_file(label_path)

            # Przekształć znormalizowane współrzędne na współrzędne piksela
            height, width, _ = image.shape
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]

            for point in pixel_points:
                cv2.drawMarker(image, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            cv2.imshow(image_filename, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """ main function """

    def train_step(image_data, target):
        """ function to apply gradients to train yolo_v3 model """

        with tf.GradientTape() as tape:
            yolo_output = yolo_v3_model(image_data)
            giou_loss = conf_loss = prob_loss = 0

            for i in range(3):
                pred_result = decode(yolo_output=yolo_output[i], num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE,
                                     classes=TRAIN_NUM_OF_CLASSES, strides=YOLO_STRIDES, anchors=YOLO_ANCHORS,
                                     index=i)
                # check_numerics(pred_result, f"pred_result[{i}]")

                loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
                                       YOLO_IOU_LOSS_THRESH)

                # check_numerics(loss_items[0], "giou_loss")
                # check_numerics(loss_items[1], "conf_loss")
                # check_numerics(loss_items[2], "prob_loss")

                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            # check_numerics(total_loss, "total_loss")

            gradients = tape.gradient(total_loss, yolo_v3_model.trainable_variables)
            # for i, grad in enumerate(gradients):
            #     check_numerics(grad, f"gradient[{i}]")

            optimizer.apply_gradients(zip(gradients, yolo_v3_model.trainable_variables))

            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))

            optimizer.learning_rate.assign(lr.numpy())

            with writer.as_default():
                tf.summary.scalar("lr", optimizer.learning_rate, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
        # wyświetlanie wartości błędów po każdym kroku
        print(
            f"Step: {global_steps.numpy()}, Total Loss: {total_loss.numpy()}, GIoU Loss: {giou_loss.numpy()}, Conf Loss: {conf_loss.numpy()}, Prob Loss: {prob_loss.numpy()}")

        return global_steps.numpy(), optimizer.learning_rate.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def validate_step(image_data, target):
        """ function to return the losses for the model during validation step """

        # obtain yolo_output from model
        yolo_output = yolo_v3_model(image_data)

        # intialise loss variables to zero
        giou_loss = conf_loss = prob_loss = 0

        # iterate over 3 scales
        for i in range(3):
            # decode resepctive yolo_output from each scale
            pred_result = decode(yolo_output=yolo_output[i], num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE,
                                 classes=TRAIN_NUM_OF_CLASSES, strides=YOLO_STRIDES, anchors=YOLO_ANCHORS,
                                 index=i)

            # compute loss with loss function
            loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
                                   YOLO_IOU_LOSS_THRESH)

            # update corresponding losses
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        # sum up losses
        total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # obtain and print list of gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')

    # if there is gpu available
    if len(gpus) > 0:

        try:

            # ensure that only necessary memory is allocated for gpu
            tf.config.experimental.set_memory_growth(gpus[0], True)

        except RuntimeError:

            pass

    # if log directory for tensorboard exist
    if os.path.exists(TRAIN_LOGDIR):
        # remove entire directory
        shutil.rmtree(TRAIN_LOGDIR)

    # creates a summary file writer training and validation for the given log directory
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    # instantiate train and test set
    trainset = Dataset(dataset_type='train', annot_path=TRAIN_ANNOT_PATH, batch_size=TRAIN_BATCH_SIZE,
                       train_input_size=TRAIN_INPUT_SIZE, strides=YOLO_STRIDES, classes_file_path=TRAIN_CLASSES,
                       anchors=YOLO_ANCHORS, anchor_per_scale=YOLO_ANCHOR_PER_SCALE,
                       max_bbox_per_scale=YOLO_MAX_BBOX_PER_SCALE, batch_frames=YOLO_BATCH_FRAMES,
                       iou_threshold=YOLO_PREPROCESS_IOU_THRESH)
    testset = Dataset(dataset_type='test', annot_path=TEST_ANNOT_PATH, batch_size=TEST_BATCH_SIZE,
                      train_input_size=TEST_INPUT_SIZE, strides=YOLO_STRIDES, classes_file_path=TRAIN_CLASSES,
                      anchors=YOLO_ANCHORS, anchor_per_scale=YOLO_ANCHOR_PER_SCALE,
                      max_bbox_per_scale=YOLO_MAX_BBOX_PER_SCALE, batch_frames=YOLO_BATCH_FRAMES,
                      iou_threshold=YOLO_PREPROCESS_IOU_THRESH)
    print("Trainset length: ", len(trainset))
    print("Validateset: ", len(testset))
    # obtain the num of steps per epoch
    steps_per_epoch = len(trainset)

    # variable to track number of steps throughout training
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

    # steps during warmup stage of training
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch

    # training steps
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # create the yolo_v3_model
    yolo_v3_model = yolo_v3(num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE, classes=TRAIN_NUM_OF_CLASSES,
                            checkpoint_dir=TRAIN_CHECKPOINTS_FOLDER, model_name=TRAIN_MODEL_NAME)

    # train from last saved checkpoint if true
    if TRAIN_FROM_CHECKPOINT:
        # load weights of last saved checkpoint
        yolo_v3_model.load_weights(yolo_v3_model.checkpoint_path).expect_partial()

    # initialise default adam optimise
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_LR_INIT)

    # initialise large best validation loss varaible to track best_val_loss
    best_val_loss = np.inf
    # Wczytaj obraz
    # image = cv2.imread(
    #     r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/validate/images/free_3/free_3_frame_0.jpg")
    # # image = cv2.imread(r"C:/Users/gosia/Downloads/0d475898-437c-428c-bc3a-c698fc039c9d.jpg")
    #
    # # Sprawdź, czy obraz został prawidłowo wczytany
    # if image is not None:
    #     print("Obraz został prawidłowo wczytany.")
    # else:
    #     print("Wystąpił problem podczas wczytywania obrazu.")
    #
    # # Wyświetl obraz
    # cv2.imshow('Obraz', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    display_images_with_crosses(images_folder, labels_folder)
    train_losses = []
    val_losses = []
    # iterate over number of epochs
    for epoch in range(TRAIN_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{TRAIN_EPOCHS}")

        # Lista do przechowywania wartości błędów w epoce
        epoch_train_losses = []
        # iterate over image and target in trainset
        for image_data, target in trainset:
            # obtain metrics from train step for given image and target
            results = train_step(image_data, target)

            total_loss = results[5]
            epoch_train_losses.append(total_loss)

            # obtain current step
            cur_step = results[0] % steps_per_epoch

            # print relevant metrics and data
            print(
                "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.9f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}".format(
                    epoch, cur_step, steps_per_epoch, results[1], results[2], results[3],
                    results[4], results[5]))

        #         if len(testset) == 0:
        #             print("configure TEST options to validate model")
        #             yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
        #             continue

        # Oblicz średni błąd dla epoki i dodaj do listy
        train_losses.append(np.mean(epoch_train_losses))
        # intialise losses for validation to zero
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0

        # iterate over valdiation testset
        for image_data, target in testset:
            # obtain losses from validation set
            results = validate_step(image_data, target)

            # update corresponding losses and count
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]

        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
        validate_writer.flush()

        # print relevant data and metrics for validation
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

        # save best validation if avg loss from current epoch is less than best known model
        if TRAIN_SAVE_BEST_ONLY and best_val_loss > total_val / count:
            # save model
            checkpoint_path = yolo_v3_model.checkpoint_path
            if not checkpoint_path.endswith('.weights.h5'):
                checkpoint_path = f"{checkpoint_path}.weights.h5"
            yolo_v3_model.save_weights(checkpoint_path)

            # update best_val_loss
            best_val_loss = total_val / count


        # save latest model
        if not TRAIN_SAVE_BEST_ONLY:
            # save model
            checkpoint_path = yolo_v3_model.checkpoint_path
            if not checkpoint_path.endswith('.weights.h5'):
                checkpoint_path = f"{checkpoint_path}.weights.h5"
            yolo_v3_model.save_weights(checkpoint_path)
        print(f"Finished epoch {epoch + 1}/{TRAIN_EPOCHS}")

        # Oblicz błąd walidacji i dodaj do listy
        val_losses.append(total_val / count)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOGDIR, 'training_validation_loss.png'))
    plt.show()

if __name__ == "__main__":
    main()

# import tensorflow as tf
# import numpy as np
# import os
# import cv2
# import inspect
# import shutil
# import sys
# from yolo_v3_model import yolo_v3
# from dataset import Dataset
# from utils import read_class_names, decode, loss_func
#
# """
# Main function to train YOLO_V3 model
# """
#
# # Train options
# TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
# TRAIN_CLASSES               = "thermographic_data/classes.txt"
# TRAIN_NUM_OF_CLASSES        = len(read_class_names(TRAIN_CLASSES))
# TRAIN_MODEL_NAME            = "model_2"
# TRAIN_ANNOT_PATH            = "thermographic_data/train"
# # TRAIN_ANNOT_PATH            = "C:/Users/gosia/OneDrive - vus.hr/Desktop/drone-detection/drone-detection/thermographic_data/train"
# TRAIN_LOGDIR                = "log" + '/' + TRAIN_MODEL_NAME
# TRAIN_CHECKPOINTS_FOLDER    = "checkpoints" + '/' + TRAIN_MODEL_NAME
# TRAIN_BATCH_SIZE            = 2
# # TRAIN_BATCH_SIZE            = 1
# TRAIN_INPUT_SIZE            = 416 # przeskalować obrazy i koordynaty bounding boxa
# # TRAIN_INPUT_SIZE            = (640, 480)  # zmień na czarno białe, albo zmniejszyć okno lub histogram i zrobić filtracje
# TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
# TRAIN_LR_INIT               = 1e-4
# # TRAIN_LR_INIT               = 1e-2
# # TRAIN_LR_INIT               = 0.001
# TRAIN_LR_END                = 1e-6
# # TRAIN_LR_END                = 1e-4
# TRAIN_WARMUP_EPOCHS         = 1
# TRAIN_EPOCHS                = 10
# TRAIN_DECAY                 = 0.8
# TRAIN_DECAY_STEPS           = 50.0
# # TRAIN_DECAY_STEPS           = 20.0
#
# # TEST options
# TEST_ANNOT_PATH             = "thermographic_data/validate"
# # TEST_ANNOT_PATH             = "C:/Users/gosia/OneDrive - vus.hr/Desktop/drone-detection/drone-detection/thermographic_data/validate"
# TEST_BATCH_SIZE             = 2
# # TEST_BATCH_SIZE             = 4 #można zmniejszyć, co spowoduje przetwarzanie mniejszej liczby obrazów naraz podczas treningu
# TEST_INPUT_SIZE             = 416
# # TEST_INPUT_SIZE             = (640, 480)
# TEST_SCORE_THRESHOLD        = 0.3
# TEST_IOU_THRESHOLD          = 0.45
#
# # YOLO options
# YOLO_STRIDES                = [8, 16, 32]
# YOLO_IOU_LOSS_THRESH        = 0.5
# YOLO_ANCHOR_PER_SCALE       = 3
# YOLO_MAX_BBOX_PER_SCALE     = 100
# # YOLO_INPUT_SIZE             = (640, 480)
# YOLO_INPUT_SIZE             = 416
# YOLO_BATCH_FRAMES           = 2
# # YOLO_BATCH_FRAMES           = 5 #można zmniejszyć liczbę klatek w sekwencji.
# YOLO_PREPROCESS_IOU_THRESH  = 0.3
# YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
#                                [[30,  61], [62,   45], [59,  119]],
#                                [[116, 90], [156, 198], [373, 326]]]
#
# def main():
#
#     """ main function """
#
#     def train_step(image_data, target):
#
#         """ function to apply gradients to train yolo_v3 model """
#
#         # record operations for automatic differentiation
#         with tf.GradientTape() as tape:
#
#             # obtain yolo_output from model
#             yolo_output = yolo_v3_model(image_data)
#
#             # intialise loss variables to zero
#             giou_loss = conf_loss = prob_loss = 0
#
#             # iterate over 3 scales
#             for i in range(3):
#
#                 # decode resepctive yolo_output from each scale
#                 pred_result = decode(yolo_output = yolo_output[i], num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE,
#                                      classes = TRAIN_NUM_OF_CLASSES, strides = YOLO_STRIDES, anchors = YOLO_ANCHORS,
#                                      index = i)
#
#                 # compute loss with loss function
#                 # loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
#                 #                        YOLO_IOU_LOSS_THRESH)
#                 loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, TRAIN_INPUT_SIZE,
#                                        YOLO_IOU_LOSS_THRESH)
#
#
#                 # update corresponding losses
#                 giou_loss += loss_items[0]
#                 conf_loss += loss_items[1]
#                 prob_loss += loss_items[2]
#
#             # sum up losses
#             total_loss = giou_loss + conf_loss + prob_loss
#
#             # computes model gradient for all trainable variables using operations recorded in context of this tape
#             gradients = tape.gradient(total_loss, yolo_v3_model.trainable_variables)
#
#             # apply model gradients to all trainable variables
#             optimizer.apply_gradients(zip(gradients, yolo_v3_model.trainable_variables))
#
#             # increment global steps
#             global_steps.assign_add(1)
#
#             # update learning rate
#             # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
#             if global_steps < warmup_steps:
#
#                 lr = global_steps / warmup_steps * TRAIN_LR_INIT
#
#             else:
#
#                 lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
#                     (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
#             # if global_steps.numpy() < 100:
#
#             #     lr = TRAIN_LR_INIT
#
#             # else:
#
#             #     lr = TRAIN_LR_INIT * pow(TRAIN_DECAY, (global_steps.numpy() / TRAIN_DECAY_STEPS))
#
#             # assign learning rate to optimizer
#             optimizer.lr.assign(lr.numpy())
#
#             # writing summary data
#             with writer.as_default():
#                 tf.summary.scalar("lr", optimizer.lr, step = global_steps)
#                 tf.summary.scalar("loss/total_loss", total_loss, step = global_steps)
#                 tf.summary.scalar("loss/giou_loss", giou_loss, step = global_steps)
#                 tf.summary.scalar("loss/conf_loss", conf_loss, step = global_steps)
#                 tf.summary.scalar("loss/prob_loss", prob_loss, step = global_steps)
#             writer.flush()
#
#         return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
#
#     def validate_step(image_data, target):
#
#         """ function to return the losses for the model during validation step """
#
#         # obtain yolo_output from model
#         yolo_output = yolo_v3_model(image_data)
#
#         # intialise loss variables to zero
#         giou_loss = conf_loss = prob_loss = 0
#
#         # iterate over 3 scales
#         for i in range(3):
#
#             # decode resepctive yolo_output from each scale
#             pred_result = decode(yolo_output = yolo_output[i], num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE,
#                                  classes = TRAIN_NUM_OF_CLASSES, strides = YOLO_STRIDES, anchors = YOLO_ANCHORS,
#                                  index = i)
#
#             # compute loss with loss function
#             # loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
#             #                        YOLO_IOU_LOSS_THRESH)
#             loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, TRAIN_INPUT_SIZE,
#                                    YOLO_IOU_LOSS_THRESH)
#
#             # update corresponding losses
#             giou_loss += loss_items[0]
#             conf_loss += loss_items[1]
#             prob_loss += loss_items[2]
#
#         # sum up losses
#         total_loss = giou_loss + conf_loss + prob_loss
#
#         return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
#
#     # obtain and print list of gpus
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print(f'GPUs {gpus}')
#
#     # if there is gpu available
#     if len(gpus) > 0:
#
#         try:
#
#             # ensure that only necessary memory is allocated for gpu
#             tf.config.experimental.set_memory_growth(gpus[0], True)
#
#         except RuntimeError:
#
#             pass
#
#     # if log directory for tensorboard exist
#     if os.path.exists(TRAIN_LOGDIR):
#
#         # remove entire directory
#         shutil.rmtree(TRAIN_LOGDIR)
#
#     # creates a summary file writer training and validation for the given log directory
#     writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
#     validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
#
#     # instantiate train and test set
#     trainset = Dataset(dataset_type = 'train', annot_path = TRAIN_ANNOT_PATH, batch_size = TRAIN_BATCH_SIZE,
#                        train_input_size = TRAIN_INPUT_SIZE, strides = YOLO_STRIDES, classes_file_path = TRAIN_CLASSES,
#                        anchors = YOLO_ANCHORS, anchor_per_scale = YOLO_ANCHOR_PER_SCALE,
#                        max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE, batch_frames = YOLO_BATCH_FRAMES,
#                        iou_threshold = YOLO_PREPROCESS_IOU_THRESH)
#     testset = Dataset(dataset_type = 'test', annot_path = TEST_ANNOT_PATH, batch_size = TEST_BATCH_SIZE,
#                       train_input_size = TEST_INPUT_SIZE, strides = YOLO_STRIDES, classes_file_path = TRAIN_CLASSES,
#                       anchors = YOLO_ANCHORS, anchor_per_scale = YOLO_ANCHOR_PER_SCALE,
#                       max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE, batch_frames = YOLO_BATCH_FRAMES,
#                       iou_threshold = YOLO_PREPROCESS_IOU_THRESH)
#     print(len(trainset))
#     print(len(testset))
#     # obtain the num of steps per epoch
#     steps_per_epoch = len(trainset)
#
#     # variable to track number of steps throughout training
#     global_steps = tf.Variable(0, trainable = False, dtype = tf.int64)
#
#     # steps during warmup stage of training
#     warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
#
#     # training steps
#     total_steps = TRAIN_EPOCHS * steps_per_epoch
#
#     # create the yolo_v3_model
#     yolo_v3_model = yolo_v3(num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, classes = TRAIN_NUM_OF_CLASSES,
#                             checkpoint_dir = TRAIN_CHECKPOINTS_FOLDER, model_name = TRAIN_MODEL_NAME)
#
#     # train from last saved checkpoint if true
#     if TRAIN_FROM_CHECKPOINT:
#
#         # load weights of last saved checkpoint
#         yolo_v3_model.load_weights(yolo_v3_model.checkpoint_path).expect_partial()
#
#     # initialise default adam optimise
#     optimizer = tf.keras.optimizers.Adam(learning_rate = TRAIN_LR_INIT)
#
#     # initialise large best validation loss varaible to track best_val_loss
#     best_val_loss = np.inf
#
#
#
#
#     # # Wczytaj obraz
#     # image = cv2.imread(r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/validate/images/free_3/free_3_frame_0.jpg")
#     # # image = cv2.imread(r"C:/Users/gosia/Downloads/0d475898-437c-428c-bc3a-c698fc039c9d.jpg")
#     #
#     #
#     # # Sprawdź, czy obraz został prawidłowo wczytany
#     # if image is not None:
#     #     print("Obraz został prawidłowo wczytany.")
#     # else:
#     #     print("Wystąpił problem podczas wczytywania obrazu.")
#     #
#     # # Wyświetl obraz
#     # cv2.imshow('Obraz', image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#
#     # class PrintSnooper:
#     #     def __init__(self, stdout):
#     #         self.stdout = stdout
#     #
#     #     def caller(self):
#     #         return inspect.stack()[2][3]
#     #
#     #     def write(self, s):
#     #         self.stdout.write("printed by %s: " % self.caller())
#     #         self.stdout.write(s)
#     #         self.stdout.write("\n")
#     #
#     #     def flush(self):
#     #         self.stdout.flush()
#     #
#     # sys.stdout = PrintSnooper(sys.stdout)
#     #
#     # iterate over number of epochs
#     for epoch in range(TRAIN_EPOCHS):
#
#         # iterate over image and target in trainset
#         for image_data, target in trainset:
#
#             # obtain metrics from train step for given image and target
#             results = train_step(image_data, target)
#
#             # obtain current step
#             cur_step = results[0] % steps_per_epoch
#
#             # print relevant metrics and data
#             print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.9f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}".format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3],
#                                              results[4], results[5]))
#
#         if len(testset) == 0:
#             print("configure TEST options to validate model")
#             filename = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME + ".weights.h5")
#
#             # Save weights using the filename
#             yolo_v3_model.save_weights(filename)
#             # yolo_v3_model.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
#             continue
#
#         # intialise losses for validation to zero
#         count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
#
#         # iterate over valdiation testset
#         for image_data, target in testset:
#
#             # obtain losses from validation set
#             results = validate_step(image_data, target)
#
#             # update corresponding losses and count
#             count += 1
#             giou_val += results[0]
#             conf_val += results[1]
#             prob_val += results[2]
#             total_val += results[3]
#
#         # writing validate summary data
#         with validate_writer.as_default():
#             tf.summary.scalar("validate_loss/total_val", total_val/count, step = epoch)
#             tf.summary.scalar("validate_loss/giou_val", giou_val/count, step = epoch)
#             tf.summary.scalar("validate_loss/conf_val", conf_val/count, step = epoch)
#             tf.summary.scalar("validate_loss/prob_val", prob_val/count, step = epoch)
#         validate_writer.flush()
#
#         # print relevant data and metrics for validation
#         print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
#               format(giou_val/count, conf_val/count, prob_val/count, total_val/count))
#
#         # save best validation if avg loss from current epoch is less than best known model
#         if TRAIN_SAVE_BEST_ONLY and best_val_loss > total_val/count:
#
#             # save model
#             # yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)
#             yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path + '.weights.h5')
#
#
#             # update best_val_loss
#             best_val_loss = total_val/count
#
#         # save latest model
#         if not TRAIN_SAVE_BEST_ONLY:
#
#             # save model
#             yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)
#         # break
#         # # Optional: add saving the model after all epochs
#         # yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)
#
# if __name__ == '__main__':
#
#     main()