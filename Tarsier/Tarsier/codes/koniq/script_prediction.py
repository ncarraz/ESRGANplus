import koniq.kutils
from koniq.kutils import model_helper as mh
from koniq.kutils import applications as apps
from koniq.kutils import image_utils as iu
from koniq.kutils import tensor_ops as ops
import pandas as pd
import numpy as np
from keras.models import Model
import argparse
import tensorflow as tf
from keras import backend as k

class Koncept512Predictor:
    def __init__(self, image_path='', aux_root="", compute_grads=False):
        self.compute_grads = compute_grads
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        k.tensorflow_backend.set_session(tf.Session(config=config))

        # build scoring model
        base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
        head = apps.fc_layers(base_model.output, name='fc',
                              fc_sizes=[2048, 1024, 256, 1],
                              dropout_rates=[0.25, 0.25, 0.5, 0],
                              batch_norm=2)

        self.model = Model(inputs=base_model.input, outputs=head)
        print('[done creating model]')

        pre = lambda im: preprocess_fn(iu.ImageAugmenter(iu.resize_image(im, (384, 512)), remap=False).fliplr(do=False).result)
        gen_params = dict(batch_size=1,
                          data_path=image_path,  # +'images/1024x768/',
                          process_fn=pre,
                          input_shape=(384, 512, 3),
                          outputs=('MOS',))

        # Wrapper for the model, helps with training and testing
        self.helper = mh.ModelHelper(self.model, 'KonCept512', pd.DataFrame(),
                                     loss='MSE', metrics=["MAE", ops.plcc_tf],
                                     monitor_metric='val_loss',
                                     monitor_mode='min',
                                     multiproc=True, workers=5,
                                     logs_root=aux_root + 'logs/koniq',
                                     models_root=aux_root + 'models/koniq',
                                     gen_params=gen_params)

        model_name = aux_root + 'models/bsz32_i1[384,512,3]_lMSE_o1[1]'
        self.grads = k.gradients(self.model.output, self.model.input)

        # uncomment to use the in-memory model
        # model_name = ''
        assert self.helper.load_model(model_name=model_name), "could not load the weights"

        print('[done initializing weights]')

    def predict(self, image, repeats=1):
        scores = []
        # ids = pd.DataFrame()
        # ids['image_name'] = [image_name]
        # ids['set'] = 'test'
        # ids['MOS'] = 1.
        for i in range(repeats):
            # generator = self.helper.make_generator(ids, shuffle=False)
            scores.append(self.model.predict(image))
            # scores.append(self.helper.predict(generator))
        # print("The scores are {}".format(np.array(scores)))
        mean_score = np.array(scores).mean(axis=0)
        if self.compute_grads:
            sess = k.get_session()
            grads = sess.run(self.grads, feed_dict={self.model.input: image})
            return mean_score[0][0], grads[0]
        return mean_score[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="path to images datset",
                        default="/Users/broz/workspaces/tests_malagan/malagan/flickr/LR/")
    parser.add_argument("--image_name", help="Image to predict",
                        default="000516_bicLRx4.png")
    parser.add_argument("--aux_root", help="Path to the koniq directory",
                        default="/Users/broz/workspaces/koniq/")
    parser.add_argument("--repeats", help="Number of times to run and average", default=1, type=int)

    args = parser.parse_args()
    print(Koncept512Predictor(args.image_path, args.aux_root).predict(args.image_name, args.repeats))
