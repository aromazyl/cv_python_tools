#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import tensorflow as tf

def warp_image(tf_image, tf_flow, width, height):
    weight_flows = tf_flow - tf.floor(tf_flow)
    floor_flows = tf.to_int32(tf.floor(tf_flow))
    floor_flat = tf.reshape(floor_flows, [-1, 2])
    floor_flows = floor_flat
    image_flat = tf.reshape(tf_image, [-1, 3])
    weight_flat = tf.reshape(weight_flows, [-1, 2])
    weight_flows = weight_flat
    x = floor_flows[:,0]
    y = floor_flows[:,1]
    xw = weight_flows[:,0]
    yw = weight_flows[:,1]
    pos_x = tf.range(height)
    pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, width])
    pos_x = tf.reshape(pos_x, [-1])
    pos_y = tf.range(width)
    pos_y = tf.tile(tf.expand_dims(pos_y, 0), [height, 1])
    pos_y = tf.reshape(pos_y, [-1])
    zero = tf.zeros([], dtype='int32')

    channels = []
    for c in range(3):
        x0 = pos_y + x
        x1 = x0 + 1
        y0 = pos_x + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, width-1)
        x1 = tf.clip_by_value(x1, zero, width-1)
        y0 = tf.clip_by_value(y0, zero, height-1)
        y1 = tf.clip_by_value(y1, zero, height-1)

        idx_a = y0 * width + x0
        idx_b = y1 * width + x0
        idx_c = y0 * width + x1
        idx_d = y1 * width + x1

        Ia = tf.gather(image_flat[:, c], idx_a)
        Ib = tf.gather(image_flat[:, c], idx_b)
        Ic = tf.gather(image_flat[:, c], idx_c)
        Id = tf.gather(image_flat[:, c], idx_d)

        wa = (1-xw) * (1-yw)
        wb = (1-xw) * yw
        wc = xw * (1-yw)
        wd = xw * yw

        img = tf.multiply(Ia, wa) + tf.multiply(Ib, wb) + tf.multiply(Ic, wc) + tf.multiply(Id, wd)
        channels.append(tf.reshape(img, shape=(height, width)))
    return tf.stack(channels, axis=-1)
    # return tf.reshape(tf.concat(channels, axis=0), shape=(height, width, 3))
    # return tf.concat(channels, axis=0)

if __name__ == '__main__':
    import image_util as iu
    import sys
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    flow = iu.GetImageFlow(iu.GetImageGray(image1), iu.GetImageGray(image2))
    with tf.Session() as sess:
        image_ph = tf.placeholder(tf.float32, shape=[480, 854, 3])
        flow_ph = tf.placeholder(tf.float32, shape=[480, 854, 2])
        result = warp_image(image_ph, flow_ph, 854, 480)
        print(flow.shape)
        image2_flow = result.eval(feed_dict = {
            image_ph : iu.ReadImage(image1),
            flow_ph : flow
            })
        print image2_flow.shape
        iu.SaveNpyImage(image2_flow, "./result.jpg")
