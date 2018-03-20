import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
from argparse import ArgumentParser
import os
try:
    import cv2
except ImportError:
    import opencv as cv2

def main():
    # parse command line arguments
    args = parse_args()
    content_is_video = args.content_type[0] == 'v'
    # make output directories if they don't exist

    for path in ('./data/','./data/outputs/', './data/frameoutputs/'):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))


    if content_is_video:
        # parameters for Shi-Tomasi feature detection
        feature_params = dict( 
            maxCorners=500,
            qualityLevel=0.001,
            minDistance=1,
            blockSize=10,
            useHarrisDetector=True,
            )

        # parameters for Lucas-Kanade optical flow algorithm
        lk_params = dict( 
            winSize=(50, 50),
            maxLevel=0,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

        # read in content video and get first frame without discarding it
        video_reader = cv2.VideoCapture(args.content_path)
        _, prev_frame = video_reader.read()
        video_reader = cv2.VideoCapture(args.content_path)

        # use a stylized version of the first frame
        prev_image = np.array([prev_frame.astype(np.float32)])
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # detect feature points
        prev_frame_points = cv2.goodFeaturesToTrack(
            prev_frame_gray, 
            mask=None, 
            **feature_params
            )

    # print out settings
    print("\n\n                      vgg: {0}".format(args.vgg_path))
    print("              style image: {0}".format(args.style_path))
    print("            content image: {0}".format(args.content_path))
    print("          update interval: {0}".format(args.update_interval))
    print("        image output path: {0}".format(args.output_path))
    print("               max_epochs: {0}".format(args.max_epochs))
    print("           content weight: {0}".format(args.content_weight))
    print("             style weight: {0}".format(args.style_weight))
    print("            learning rate: {0}".format(args.learning_rate))

    if content_is_video:
        print("          temporal weight: {0}".format(args.temporal_weight))
        print("             block length: {0}".format(args.block_length))
        print("              video image: {0}".format(args.content_path))
        print("         frame input path: {0}".format(args.frame_input_path.format("<frame_index>")))
        print("        frame output path: {0}".format(args.frame_output_path.format("<frame_index>")))
        print("         epoch input path: {0}".format(args.epoch_input_path.format("<frame_index>")))
        print("        epoch output path: {0}\n\n".format(args.epoch_output_path.format("<frame_index>","<epoch_index>")))
    

    cnn_factory = CNNFactory(args.vgg_path)
    style_image = np.array([scipy.misc.imread(args.style_path).astype(np.float32)])
    frame_index = 0
    continuing = True
    
    while continuing:
        # get next content image
        if content_is_video:
            frame_returned = False
            while not frame_returned:
                # read in the next content_image, if it exists
                frame_returned, content_image = video_reader.read()
                if not frame_returned:
                    content_image = None
                    continuing = False
                    break
                
                curr_frame_gray = cv2.cvtColor(content_image, cv2.COLOR_BGR2GRAY)
                # skip current content_image if too similar to previous content_image
                if np.count_nonzero(curr_frame_gray - prev_frame_gray) < 5000 and frame_index > 0:
                    frame_returned = False
                    continue

                scipy.misc.imsave(args.frame_input_path.format(frame_index), content_image)
                scipy.misc.imsave(args.epoch_input_path.format(frame_index), content_image)

                # compute optical flow
                curr_frame_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_frame_points, None, **lk_params)
                curr_good_points = curr_frame_points[st==1]

                # matrix with elements marked as 0 in disoccluded regions, 1 elsewise
                content_image = np.array([content_image.astype(np.float32)])
                occlusion = np.ones(content_image.shape, dtype=np.float32)
                for _, curr_point in enumerate(curr_good_points):
                    a, b = curr_point.ravel()
                    for row in range(max(0, int(a - args.block_length / 2)), min(occlusion.shape[1], int(a + (args.block_length + 1) / 2))):
                        for col in range(max(0, int(b - args.block_length / 2)), min(occlusion.shape[2], int(b + (args.block_length + 1) / 2))):
                            occlusion[0][row][col][:] = np.zeros(content_image.shape[3])

                print("\nframe: {0}".format(str(frame_index + 1)))

        elif not content_is_video:
            continuing = False
            content_image = np.array([scipy.misc.imread(args.content_path).astype(np.float32)])
       
        if content_image is None:
            break

        g = tf.Graph()
        with g.as_default(), tf.Session() as sess:
            # compute content features
            content_CNN = cnn_factory.newConvNet(content_image)
            content_layer = 'relu4_2'
            target_content = content_CNN[content_layer].eval()
            
            # compute style features
            style_CNN = cnn_factory.newConvNet(style_image)
            target_style = {}
            style_layers = ['relu{0}_1'.format(i) for i in range(1,6)]
            for layer in style_layers:
                # compute gram matrix
                style_layer = style_CNN[layer].eval()
                style_layer = np.reshape(style_layer, (-1, style_layer.shape[3]))
                target_style[layer] = np.matmul(np.transpose(style_layer), style_layer) / style_layer.size
            
            if frame_index == 0:
                occlusion = np.zeros(content_image.shape, dtype=np.float32)
                
            # intialize the output image
            if frame_index == 0 and args.initial_path != 0:
                initial_image = np.array([scipy.misc.imread(args.initial_path).astype(np.float32)])
            else:
                initial_image = tf.truncated_normal(content_image.shape, mean=250, stddev=3.0) / 256.0
            output_image = tf.Variable(initial_image)
            output_CNN = cnn_factory.newConvNet(output_image)
            losses = []
            total_loss = 0
            style_loss = 0

            # compute content loss
            content_loss = args.content_weight * .5 * tf.nn.l2_loss(output_CNN[content_layer] - target_content) /  target_content.size
            losses.append(content_loss)

            # compute style loss
            for layer in style_layers:
                style_layer = output_CNN[layer]
                _, height, width, number = map(lambda i: i.value, style_layer.get_shape())
                num_values = height * width * number
                target_content = tf.reshape(style_layer, (-1, number))
                neural_style = tf.matmul(tf.transpose(target_content), target_content) / num_values
                original_style = target_style[layer]
                style_loss += args.style_weight * .25 * tf.nn.l2_loss(original_style - neural_style) / original_style.size
            losses.append(style_loss)

            if frame_index != 0 and content_is_video:
                # compute temporal consistency loss
                temporal_loss = tf.reduce_sum(tf.mul(occlusion, tf.nn.l2_loss(output_image - prev_image))) / content_image.size
                temporal_loss = args.temporal_weight * tf.cast(temporal_loss, tf.float32)
                losses.append(temporal_loss)

            # define total loss as the sum of all losses
            for i in range(len(losses)):
                total_loss += losses[i]
            
            # optimize over total loss
            optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)
            best_loss = float('inf')

            with tf.Session() as sess:
                # initialize TensorFlow variables
                sess.run(tf.global_variables_initializer())

                # run training max_epochs
                for epoch in range(int(args.initial_epoch), int(args.max_epochs) + 1):
                    # print epoch number

                    #TODO: cleanup
                    epoch_string = str(epoch)
                    if epoch < 10:
                        epoch_string = '0' + epoch_string
                    if epoch < 100:
                        epoch_string = '0' + epoch_string
                    print('    epoch: {0}/{1}'.format(epoch_string, args.max_epochs))

                    # compute gradients and apply
                    optimizer.run()

                    if epoch % args.update_interval == 0:
                        # evaluate losses
                        curr_losses = []
                        curr_total_loss = 0

                        # print losses
                        print("")
                        for i in range(len(losses)):
                            curr_losses.append(losses[i].eval())
                            curr_total_loss += curr_losses[i]
                            loss_strings = [' content loss: {0}', '   style loss: {0}', 'temporal loss: {0}']
                            print(loss_strings[i].format(curr_losses[i]))
                        print("")

                        # update best
                        if curr_total_loss < best_loss:
                            best_loss = curr_total_loss
                            best_img = output_image.eval()

                        # produce output image and save
                        out = np.clip(best_img.reshape(content_image.shape[1:]), 0, 255).astype(np.uint8)
                        scipy.misc.imsave(args.output_path.format(epoch), out)

                        # save output image on final epoch
                        if epoch == args.max_epochs:
                            scipy.misc.imsave(args.output_path.format(""), out)
                            frame_index += 1

                            if content_is_video:
                                prev_image = out
                                prev_frame_points = curr_frame_points
            

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--content_type',
        help='(v)ideo or (i)mg',
        default='img')
    parser.add_argument(
        '--content_path',
        help='input content path',
        default='data/content.jpg')
    parser.add_argument(
        '--style_path',
        help='style path',
        default='data/style.jpg')
    parser.add_argument(
        '--vgg_path',
        help='vgg path',
        default='data/imagenet-vgg-verydeep-19.mat')
    parser.add_argument(
        '--frame_input_path',
        help='frame input save path',
        default='data/frameoutputs/input{0}.jpg')
    parser.add_argument(
        '--frame_output_path',
        help='frame output path',
        default='data/frameoutputs/output{0}.jpg')
    parser.add_argument(
        '--output_path',
        help='output path',
        default='data/outputs/output{0}.jpg')
    parser.add_argument(
        '--epoch_input_path',
        help='epoch input save path',
        default='data/frameoutputs/outputs{0}/input.jpg')
    parser.add_argument(
        '--epoch_output_path',
        help='epoch output path',
        default='data/frameoutputs/outputs{0}/output{1}.jpg')
    parser.add_argument(
        '--initial_path',
        help='initial path',
        default=0)
    parser.add_argument(
        '--initial_epoch',
        help='initial epoch',
        default=0)
    parser.add_argument(
        '--update_interval',
        help='how often to save and print out losses',
        type=int,
        default=1)
    parser.add_argument(
        '--learning_rate',
        help='learning rate',
        type=float,
        default=5)
    parser.add_argument(
        '--max_epochs',
        help='max_epochs',
        type=int,
        default=2500)
    parser.add_argument(
        '--content_weight',
        help='content weight',
        type=float,
        default=10)
    parser.add_argument(
        '--style_weight',
        help='style weight',
        type=float,
        default=250)
    parser.add_argument(
        '--temporal_weight',
        help='temporal weight',
        type=float,
        default=1e-2)
    parser.add_argument(
        '--block_length',
        help='block length',
        type=int,
        default=100)
    return parser.parse_args()

class CNNFactory:

    def __init__(self, data_path):
        self.data = scipy.io.loadmat(data_path)

    def newConvNet(self, image):
        weights = self.data['layers'][0]
        net = {}
        curr = image
        for i in range(len(weights)):
            layer = weights[i][0][0][0][0]
            type = layer[:4]
            if type == 'conv':
                kernels, bias = weights[i][0][0][2][0]
                curr = net[layer] = self.__newConvLayer(
                    curr,
                    np.transpose(kernels, (1, 0, 2, 3)),
                    bias.reshape(-1))
            elif type == 'pool':
                curr = net[layer] = self.__newPoolingLayer(curr)
            elif type == 'relu':
                curr = net[layer] = self.__newReluLayer(curr)
        return net

    def __newConvLayer(self, input, kernels, bias):
        conv = tf.nn.conv2d(
            input,
            tf.constant(kernels),
            strides=(1, 1, 1, 1),
            padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def __newPoolingLayer(self, input):
        return tf.nn.max_pool(
            input,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME')

    def __newReluLayer(self, input):
        return tf.nn.relu(input)

if __name__ == "__main__":
    main()
