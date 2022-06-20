from utils.model_utils import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
from enum import IntEnum


class ModifiedLossMode(IntEnum):
    NoModified = 1
    EqualWeights = 2
    UnequalWeights = 3


class DepthEstimationModel(tf.keras.Model):
    
    def __init__(self, ssim_weight=0.85, l1_weight=0.1, edge_weight=0.9, image_size=256, modified_loss=ModifiedLossMode.NoModified):
        super().__init__()
        
        self.ssim_loss_weight = ssim_weight
        self.l1_loss_weight = l1_weight
        self.edge_loss_weight = edge_weight
        
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.rmse_metric = tf.keras.metrics.Mean(name='rmse')
        self.acc_1_metric = tf.keras.metrics.Mean(name='acc_1.25')
        self.acc_2_metric = tf.keras.metrics.Mean(name='acc_2_1.25')

        self.image_size = image_size
        self.modified_loss = modified_loss

        f = [16, 32, 64, 128, 256]
        
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
        ]
        
        self.bottle_neck_block = BottleNeckBlock(f[4])
      
        self.upscale_blocks = [
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]
        
        self.conv_layer = keras.layers.Conv2D(1, (1, 1), padding="same", activation="linear")


    def calculate_error_metrics(self, target, pred):
        const = tf.constant(1e-8)
        rmse = 0
        a1 = 0 
        a2 = 0 
        batch_size = target.shape[0]

        for target_i, pred_i in zip(target, pred):
            
            thresh = tf.maximum((target_i / (pred_i + const)), (pred_i / (target_i + const)))
            a1_i = tf.reduce_mean(tf.cast(thresh < 1.25, tf.float32))
            a2_i = tf.reduce_mean(tf.cast(thresh < (1.25 ** 2), tf.float32))

            se_i = (target_i - pred_i) ** 2
            rmse_i = tf.math.sqrt(tf.reduce_mean(se_i))

            rmse += rmse_i
            a1 += a1_i
            a2 += a2_i

        return rmse/batch_size, a1/batch_size, a2/batch_size


    def calculate_loss(self, target, pred):
        # Edges
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

        # Structural similarity (SSIM) index
        ssim_index = tf.image.ssim(target, pred, max_val=self.image_size, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2)
        ssim_loss = tf.reduce_mean(1 - ssim_index)

        # Point-wise depth
        l1_loss = tf.reduce_mean(tf.abs(target - pred))

        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
        )

        return loss


    @property
    def metrics(self):
        return [self.loss_metric, self.rmse_metric, self.acc_1_metric, self.acc_2_metric]


    def train_step(self, batch_data):
        input, target = batch_data
        
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)
            rmse, a1, a2 = self.calculate_error_metrics(target, pred)

            if self.modified_loss == ModifiedLossMode.EqualWeights:
                loss = 0.5*loss + 0.5*loss
            elif self.modified_loss == ModifiedLossMode.UnequalWeights:
                loss = 0.3*loss + 0.7*rmse

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        
        self.rmse_metric.update_state(rmse)
        self.acc_1_metric.update_state(a1)
        self.acc_2_metric.update_state(a2)

        return {"loss": self.loss_metric.result(), 'rmse' : self.rmse_metric.result(), 'acc_1.25' : self.acc_1_metric.result(), 
            'acc_2_1.25' : self.acc_2_metric.result()}


    def test_step(self, batch_data):
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)
        self.loss_metric.update_state(loss)

        rmse, a1, a2 = self.calculate_error_metrics(target, pred)
        self.rmse_metric.update_state(rmse)
        self.acc_1_metric.update_state(a1)
        self.acc_2_metric.update_state(a2)

        return {"loss": self.loss_metric.result(), 'rmse' : self.rmse_metric.result(), 'acc_1.25' : self.acc_1_metric.result(), 
            'acc_2_1.25' : self.acc_2_metric.result()}


    def call(self, data):
        frames = data[..., :3]
        masks = tf.expand_dims(data[..., 3], axis=-1)

        c1, p1 = self.downscale_blocks[0](frames)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4) * masks