import datetime

import matplotlib.pyplot as plt
from keras.src.layers import BatchNormalization
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import Input, backend as K, Model
from tensorflow.python.keras.layers import UpSampling2D, Conv2D, Dropout, Concatenate, LeakyReLU, Dense, Reshape, \
    Lambda, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# when the conditioning reproduction weight is set to be 0, it becomes to be GAN
weight_JS = 1  # weight for Jensen Shannon divergence
weight_well = 2  # weight for well data
weight_seismic = 1  # weight for seismic
epochs = 600
weight_dir = './weight/'
img_dir = './img/'
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class StochasticPix2pix:
    def __init__(self):
        # Input param
        self.batch_size = 8
        # training,output image shape
        self.img_rows = 56
        self.img_cols = 56
        # # of latent variables
        self.latent = 20
        # # of facies
        self.channels = 5
        # input and output image shape
        self.img_shape1 = (self.img_rows, self.img_cols, 1)  # B seismic outline
        self.img_shape2 = (self.img_rows, self.img_cols, self.channels)  # A training,output image shape
        # well location
        self.well_loc = np.array(
            [[28, 28], [1, 4], [32, 29], [33, 33], [12, 13], [4, 46], [44, 40], [20, 40], [51, 15], [28, 47]])
        # regions corresponding to the largest template size
        # Note we assume nonstationarity (regional stationary) if they are larger than 1
        self.disc_patch = 1
        # number of filters in the first layer of G and D, D's filter size can be inferred from the MPS density function.
        self.gf = 20
        self.df = 5
        # weight optimization param

        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=epochs,
            decay_rate=0.9)
        optimizer = Adam(lr_schedule, 0.9, amsgrad=True)


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images (params)
        img_B = Input(shape=self.img_shape1)  # seismic outline
        well = Input(shape=self.img_shape1)  # well data map
        random = Input(shape=(self.latent,))  # random latent variables

        # Generate conditional realization fake_a, reproduced well data, and seismic outline with generator
        fake_a, fake_well, fake_seismic = self.generator([img_B, random, well])

        # to train the generator to fool the discriminator we will only train the generator
        self.discriminator.trainable = False

        # Discriminator calculates the multipoint statistics and determines if the statistics calculated
        # from training images are reproduced in the realization.
        valid = self.discriminator(fake_a)

        # specify the relative weight for each loss functions
        self.combined = Model(inputs=[img_B, random, well], outputs=[valid, fake_well, fake_seismic])
        self.combined.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
                              loss_weights=[weight_JS, weight_well, weight_seismic],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during down sampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during up sampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input (seismic)
        d0 = Input(shape=self.img_shape1)
        # latent variables
        random_input = Input(shape=(self.latent,))
        random_layer = Dense(self.img_rows * self.img_cols)(random_input)
        random_layer = Reshape((self.img_rows, self.img_cols, 1))(random_layer)
        # combine latent variables with seismic input
        combine = Concatenate()([d0, random_layer])
        # input well data
        well = Input(shape=self.img_shape1)
        # combine all 3 inputs
        # combine=Concatenate(axis=-1)([well,combine])
        combine = Concatenate()([well, combine])
        # Encoder-Decoder Structure
        d1 = conv2d(combine, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        # Start of Decoding
        u8 = deconv2d(d3, d2, self.gf * 2)
        u9 = deconv2d(u8, d1, self.gf)
        u10 = UpSampling2D(size=2)(u9)
        u11 = Concatenate()([u10, well])
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='softmax')(u11)

        # Back calculation of seismic data
        seismic = Lambda(lambda x: x[:, :, :, 1:])(output_img)
        seismic = Lambda(lambda x: K.sum(x, axis=-1), name='seismic')(seismic)

        # Back calculation of well data
        wells = [Lambda(lambda x: x[:, 13:14, 40, :])(output_img), Lambda(lambda x: x[:, 19:20, 4, :])(output_img),
                 Lambda(lambda x: x[:, 32:33, 29, :])(output_img), Lambda(lambda x: x[:, 33:34, 33, :])(output_img),
                 Lambda(lambda x: x[:, 12:13, 13, :])(output_img), Lambda(lambda x: x[:, 4:5, 46, :])(output_img),
                 Lambda(lambda x: x[:, 44:45, 40, :])(output_img), Lambda(lambda x: x[:, 20:21, 40, :])(output_img),
                 Lambda(lambda x: x[:, 51:52, 15, :])(output_img), Lambda(lambda x: x[:, 28:29, 47, :])(output_img)]

        wells = Concatenate(axis=1)(wells)

        return Model(inputs=[d0, random_input, well], outputs=[output_img, wells, seismic])

    def build_discriminator(self):
        # define convolutional down sampling layer
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # input training image/output from generator
        img_A = Input(shape=self.img_shape2)

        # MPS extraction
        d1 = d_layer(img_A, self.df, bn=False)  # 128
        d2 = d_layer(d1, self.df * 2)  # 64
        d3 = d_layer(d2, self.df * 4)  # 32
        # Force output between 0 and 1
        #        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d3)
        d3 = Flatten()(d3)
        validity = Dense(1, activation='sigmoid')(d3)
        return Model(img_A, validity)

    def train(self, epochs):
        summary_writer = tf.summary.create_file_writer(log_dir)
        # Adversarial loss ground truths
        # batch_size = self.batch_size
        #        valid = np.ones((batch_size,) + self.disc_patch)
        #        fake = np.zeros((batch_size,) + self.disc_patch)
        # valid = np.ones(batch_size)
        # fake = np.zeros(batch_size)
        # load training images      
        imgs = np.load('train_dat.npy')
        imgs_B_all = imgs.copy()
        imgs_B_all[imgs <= 0] = 0
        imgs_B_all[imgs > 0] = 1
        imgs_A_all = imgs[:, :, :, 0].copy()

        batch_size = imgs_B_all.shape[0]
        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)

        well_all = np.zeros(imgs_B_all.shape)
        for mm in range(self.well_loc.shape[0]):
            well_all[:, self.well_loc[mm, 0], self.well_loc[mm, 1], 0] = (imgs_A_all[:, self.well_loc[mm, 0],
                                                                          self.well_loc[mm, 1]]) / 2  # input
        # well_dat_all well data from loss calculation
        well_dat_all = np.zeros((imgs_B_all.shape[0], self.well_loc.shape[0]))
        for mm in range(self.well_loc.shape[0]):
            well_dat_all[:, mm] = imgs_A_all[:, self.well_loc[mm, 0], self.well_loc[mm, 1]]  # output
        well_dat_all = to_categorical(well_dat_all, self.channels)
        # one-hot encoding for training images
        imgs_A_all = to_categorical(imgs_A_all, self.channels)
        ids = np.arange(imgs_B_all.shape[0])

        # res_all = np.random.normal(0, 1, size=(imgs_B_all.shape[0], self.latent))

        # start of training
        for jj in range(epochs):

            # imgs_A = imgs_A_all[ids[int(1 * batch_size):int(2 * batch_size)]]
            # imgs_B = imgs_B_all[ids[int(1 * batch_size):int(2 * batch_size)]]
            # well = well_all[ids[int(1 * batch_size):int(2 * batch_size)]]
            # self.generate_images(imgB=imgs_B[0:1, :, :, :], imgA=imgs_A[0], well=well[0:1])
            #
            # # for step in range(batch_size//2):
            # # generate random latent input variables
            # res_all = np.random.normal(0, 1, size=(imgs_B_all.shape[0], self.latent))
            #
            # np.random.shuffle(ids)
            # for ii in np.arange(imgs_B_all.shape[0] // batch_size):
            #     # divide training data according to their batch size
            #     imgs_A = imgs_A_all[ids[int(ii * batch_size):int((ii + 1) * batch_size)]]
            #     imgs_B = imgs_B_all[ids[int(ii * batch_size):int((ii + 1) * batch_size)]]
            #     well_dat = well_dat_all[ids[int(ii * batch_size):int((ii + 1) * batch_size)]]
            #     well = well_all[ids[int(ii * batch_size):int((ii + 1) * batch_size)]]
            #     res = res_all[ids[int(ii * batch_size):int((ii + 1) * batch_size)]]
            #
            #     # ---------------------
            #     #  Train Discriminator
            #     # ---------------------
            #     # generate fake realization and its associated conditioning data
            #     fake_A, fake_well, fake_seismic = self.generator.predict([imgs_B, res, well])
            #
            #     # train discriminator to calculate J-S divergence
            #     d_loss_real = self.discriminator.train_on_batch(imgs_A, valid)
            #     d_loss_fake = self.discriminator.train_on_batch(fake_A, fake)
            #     d_loss = 1 * np.add(d_loss_real, d_loss_fake)
            #
            #     # -----------------
            #     #  Train Generator
            #     # -----------------
            #
            #     # Train the generators
            #     g_loss = self.combined.train_on_batch([imgs_B, res, well], [valid, well_dat, imgs_B[:, :, :, 0]])
            #
            #     fake_well = np.argmax(fake_well, axis=-1)
            #     well_dat2 = np.argmax(well_dat, axis=-1)
            #     error = np.sum((fake_well != well_dat2)) / fake_well.shape[0]
            #
            #     with summary_writer.as_default():
            #         tf.summary.scalar('d_loss', d_loss[0], step=jj + 1)
            #         tf.summary.scalar('d_acc', 100 * d_loss[1], step=jj + 1)
            #         tf.summary.scalar('g_loss', g_loss[0], step=jj + 1)
            #         tf.summary.scalar('well error', error * 10, step=jj + 1)
            #
            #     # output the training progress
            #     if ii % 20 == 0:
            #         # calculate well data mismatch
            #         # print("[Epoch %d,%d (step: %d)] [D loss: %f, D acc: %3d%%] [G loss: %f][well error:%3d%%] " % (
            #         # jj + 1, epochs, step + 1,
            #         # d_loss[0], 100 * d_loss[1],
            #         # g_loss[0], error * 10))
            #
            #         print("[Epoch %d,%d] [D loss: %f, D acc: %3d%%] [G loss: %f][well error:%3d%%] " % (
            #             jj + 1, epochs,
            #             d_loss[0], 100 * d_loss[1],
            #             g_loss[0], error * 10))
            #
            #     # save generated image samples
            #     if ii == 0:
            #         self.save_imgs(jj, imgB=imgs_B[0:1, :, :, :], imgA=imgs_A[0], well=well[0:1])
            #     # save weight and training process
            # imgs_A = imgs_A_all[ids[int(1 * batch_size):int(2 * batch_size)]]
            # imgs_B = imgs_B_all[ids[int(1 * batch_size):int(2 * batch_size)]]
            # well = well_all[ids[int(1 * batch_size):int(2 * batch_size)]]

            res_all = np.random.normal(0, 1, size=(imgs_B_all.shape[0], self.latent))
            np.random.shuffle(ids)

            B = imgs_B_all[ids]
            A = imgs_A_all[ids]
            W = well_all[ids]
            D = well_dat_all[ids]
            R = res_all[ids]

            for ii in np.arange(B.shape[0] // batch_size):
                part_B = B[batch_size * ii:batch_size * (ii + 1)]
                part_A = A[batch_size * ii:batch_size * (ii + 1)]
                part_W = W[batch_size * ii:batch_size * (ii + 1)]
                part_D = D[batch_size * ii:batch_size * (ii + 1)]
                part_R = R[batch_size * ii:batch_size * (ii + 1)]

                # generate fake realization and its associated conditioning data
                fake_A, fake_well, fake_seismic = self.generator.predict([part_B, part_R, part_W])

                # train discriminator to calculate J-S divergence
                d_loss_real = self.discriminator.train_on_batch(part_A, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_A, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generators
                g_loss = self.combined.train_on_batch([part_B, part_R, part_W], [valid, part_D, part_B[:, :, :, 0]])

                fake_well = np.argmax(fake_well, axis=-1)
                well_dat2 = np.argmax(part_D, axis=-1)
                error = np.sum((fake_well != well_dat2)) / fake_well.shape[0]

                with summary_writer.as_default():
                    tf.summary.scalar('d_loss', d_loss[0], step=jj + 1)
                    tf.summary.scalar('d_acc', 100 * d_loss[1], step=jj + 1)
                    tf.summary.scalar('g_loss', g_loss[0], step=jj + 1)
                    tf.summary.scalar('well error', error * 10, step=jj + 1)

                print("[Epoch %d,%d (batch: %d)] [D loss: %f, D acc: %3d%%] [G loss: %f][well error:%3d%%] " % (
                    jj + 1, epochs, ii + 1,
                    d_loss[0], 100 * d_loss[1],
                    g_loss[0], error * 10))

            self.generate_images(imgB=B[0:1, :, :, :], imgA=A[0], well=W[0:1])

            # save generated image samples
            self.save_imgs(jj, imgB=B[0:1, :, :, :], imgA=A[0], well=W[0:1])

        # self.generator.save_weights(weight_dir + 'pos_predictor_full_seismic_filt2_sand_whole2.weights.h5')
        # self.combined.save_weights(weight_dir + 'pos_combined_full_seismic_filt2_sand_whole2.weights.h5')
        # self.discriminator.save_weights(weight_dir + 'pos_discriminator_full_seismic_filt2_sand_whole2.weights.h5')

    def generate_images(self, imgA, imgB, well):
        plt.close('all')
        plt.figure(figsize=(8, 4))

        zz = np.argmax(imgA[:, :, :], axis=-1)
        plt.subplot(1, 4, 1)
        plt.imshow(np.argmax(imgA[:, :, :], axis=-1))
        plt.scatter(self.well_loc[:, 0], self.well_loc[:, 1], color='red')
        plt.title('Truth Model')
        plt.axis('off')

        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        plt.subplot(1, 4, 2)
        plt.imshow(np.argmax(fake_A[0, :, :, :], axis=-1))
        plt.title('Realization1')
        plt.axis('off')
        # realization 2

        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        plt.subplot(1, 4, 3)
        plt.imshow(np.argmax(fake_A[0, :, :, :], axis=-1))
        plt.title('Realization2')
        plt.axis('off')

        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        z = np.argmax(fake_A[0, :, :, :], axis=-1)
        z2 = np.ones(z.shape) * np.nan
        for i in range(self.well_loc.shape[0]):
            z2[self.well_loc[i, 1], self.well_loc[i, 0]] = (
                    z[self.well_loc[i, 0], self.well_loc[i, 1]] == zz[self.well_loc[i, 0], self.well_loc[i, 1]])
        plt.subplot(1, 4, 4)
        plt.imshow(z2)
        plt.title('Well Data Mismatch')
        plt.axis('off')
        plt.show()

    def save_imgs(self, ii, imgB, imgA, well):
        # training image
        plt.figure()
        zz = np.argmax(imgA[:, :, :], axis=-1)
        plt.imshow(np.argmax(imgA[:, :, :], axis=-1))
        plt.scatter(self.well_loc[:, 0], self.well_loc[:, 1], color='red')
        plt.title('Truth Model')
        plt.savefig(img_dir + str(ii) + '_true.png', dpi=100)
        plt.close('all')
        # realization 1
        plt.figure()
        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        plt.imshow(np.argmax(fake_A[0, :, :, :], axis=-1))
        plt.title('Realization1')
        plt.savefig(img_dir + str(ii) + '_realization1.png', dpi=100)
        plt.close('all')
        # realization 2
        plt.figure()
        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        z = np.argmax(fake_A[0, :, :, :], axis=-1)
        plt.imshow(z)
        plt.title('Realization2')
        plt.savefig(img_dir + str(ii) + '_realization2.png', dpi=100)
        plt.close('all')
        # wel data mismatch map
        plt.figure()
        res = np.random.normal(0, 1, size=[1, self.latent])
        fake_A, fake_well, seismic = self.generator.predict([imgB, res, well])
        z = np.argmax(fake_A[0, :, :, :], axis=-1)
        z2 = np.ones(z.shape) * np.nan
        for i in range(self.well_loc.shape[0]):
            z2[self.well_loc[i, 1], self.well_loc[i, 0]] = z[self.well_loc[i, 0], self.well_loc[i, 1]] == zz[
                self.well_loc[i, 0], self.well_loc[i, 1]]
        plt.imshow(z2)
        plt.title('Well Data Mismatch')
        plt.savefig(img_dir + str(ii) + '_well_dat.png', dpi=100)
        plt.close('all')
        return 0


if __name__ == '__main__':
    gan = StochasticPix2pix()

    # gan.generator.load_weights(f'{weight_dir}pos_predictor_full_seismic_filt2_sand_whole2_{epochs-1}.weights.h5')
    # gan.discriminator.load_weights(f'{weight_dir}pos_discriminator_full_seismic_filt2_sand_whole2_{epochs-1}.weights.h5')
    # gan.combined.load_weights(f'{weight_dir}pos_combined_full_seismic_filt2_sand_whole2_{epochs-1}.weights.h5')
    gan.train(epochs)

