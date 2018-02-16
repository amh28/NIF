# -*- coding: utf-8 -*-
"""
This module defines a general procedure for running applications.

Example usage::
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()

``system_param`` and ``input_data_param`` should be generated using:
``niftynet.utilities.user_parameters_parser.run()``
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.application_variables import \
    GradientsCollector, OutputsCollector, global_vars_init_or_restore
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.image_sets_partitioner import TRAIN, VALID, INFER
from niftynet.io.misc_io import get_latest_subfolder, touch_folder
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities.util_common import set_cuda_device

FILE_PREFIX = 'model.ckpt'


class ApplicationDriver(object):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session and coordinator. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting ``niftynet.application.base_application``
    to be compatible with this driver.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.app = None
        self.graph = tf.Graph()

        self.saver = None

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.summary_dir = None
        self.session_prefix = None
        self.max_checkpoints = 2
        self.save_every_n = 10
        self.tensorboard_every_n = -1

        self.validation_every_n = -1
        self.validation_max_iter = 1

        self.initial_iter = 0
        self.final_iter = 0

        self._coord = None
        self._init_op = None
        self._data_partitioner = None
        self.outputs_collector = None
        self.gradients_collector = None
        self.GROUNDTRUTH = None
        self.PREDICTION = None
        self.CONT = 0
        self.GT = None
        self.P = None
        self.MESSAGE = None
        self.INI_WEIGHT = None
        
    def initialise_application(self, workflow_param, data_param):
        """
        This function receives all parameters from user config file,
        create an instance of application.

        :param workflow_param: a dictionary of user parameters,
            keys correspond to sections in the config file
        :param data_param: a dictionary of input image parameters,
            keys correspond to data properties to be used by image_reader
        :return:
        """
        try:
            system_param = workflow_param.get('SYSTEM', None)
            net_param = workflow_param.get('NETWORK', None)
            train_param = workflow_param.get('TRAINING', None)
            infer_param = workflow_param.get('INFERENCE', None)
            app_param = workflow_param.get('CUSTOM', None)
        except AttributeError:
            tf.logging.fatal('parameters should be dictionaries')
            raise

        assert os.path.exists(system_param.model_dir), \
            'Model folder not exists {}'.format(system_param.model_dir)
        self.is_training = (system_param.action == "train")
        # hardware-related parameters
        self.num_threads = max(system_param.num_threads, 1) \
            if self.is_training else 1
        self.num_gpus = system_param.num_gpus \
            if self.is_training else min(system_param.num_gpus, 1)
        set_cuda_device(system_param.cuda_devices)

        # set output TF model folders
        self.model_dir = touch_folder(
            os.path.join(system_param.model_dir, 'models'))
        self.session_prefix = os.path.join(self.model_dir, FILE_PREFIX)

        if self.is_training:
            assert train_param, 'training parameters not specified'
            summary_root = os.path.join(system_param.model_dir, 'logs')
            self.summary_dir = get_latest_subfolder(
                summary_root,
                create_new=train_param.starting_iter == 0)

            self.initial_iter = train_param.starting_iter
            self.final_iter = max(train_param.max_iter, self.initial_iter)
            self.save_every_n = train_param.save_every_n
            self.tensorboard_every_n = train_param.tensorboard_every_n
            self.max_checkpoints = \
                max(train_param.max_checkpoints, self.max_checkpoints)
            self.gradients_collector = GradientsCollector(
                n_devices=max(self.num_gpus, 1))
            self.validation_every_n = train_param.validation_every_n
            if self.validation_every_n > 0:
                self.validation_max_iter = max(self.validation_max_iter,
                                               train_param.validation_max_iter)
            action_param = train_param
        else:
            assert infer_param, 'inference parameters not specified'
            self.initial_iter = infer_param.inference_iter
            action_param = infer_param

        self.outputs_collector = OutputsCollector(
            n_devices=max(self.num_gpus, 1))

        # create an application instance
        assert app_param, 'application specific param. not specified'
        # app_param.name = netSegment
        app_module = ApplicationDriver._create_app(app_param.name)


        #NET_PARAM: activation_function=u'prelu', batch_size=1, bias_initializer='zeros', cutoff=(0.01, 0.99),
        # decay=0, foreground_type=u'otsu_plus', histogram_ref_file=u'/home/alemh/NIF/trainingab_MODEL/histogram1.txt',
        # multimod_foreground_type=u'and', name=u'dense_vnet', norm_type='percentile', normalisation=True, 
        # normalise_foreground_only=True, queue_length=8, reg_type='L2', volume_padding_size=(0, 0, 0), 
        # weight_initializer='he_normal', whitening=True, window_sampling=u'resize')

        #ACTION_PARAM: Namespace(exclude_fraction_for_inference=0.0, exclude_fraction_for_validation=0.0, 
        #loss_type=u'Dice',
        # lr=0.0005, max_checkpoints=50, max_iter=2, optimiser='adam', random_flipping_axes=(1,), 
        # rotation_angle=(-10.0, 10.0), rotation_angle_x=(), rotation_angle_y=(), rotation_angle_z=(), 
        # sample_per_volume=4, save_every_n=500, scaling_percentage=(-10.0, 10.0), starting_iter=0, 
        # tensorboard_every_n=20, validation_every_n=-1, validation_max_iter=1)


        self.app = app_module(net_param, action_param, self.is_training)

        # initialise data input
        data_partitioner = ImageSetsPartitioner()
        # clear the cached file lists
        data_partitioner.reset()
        do_new_partition = \
            self.is_training and self.initial_iter == 0 and \
            (not os.path.isfile(system_param.dataset_split_file)) and \
            (train_param.exclude_fraction_for_validation > 0 or
             train_param.exclude_fraction_for_inference > 0)
        data_fractions = None
        if do_new_partition:
            assert train_param.exclude_fraction_for_validation > 0 or \
                   self.validation_every_n <= 0, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available,\nplease ' \
                'check "exclude_fraction_for_validation" in the config ' \
                'file (current config value: {}).'.format(
                    self.validation_every_n,
                    train_param.exclude_fraction_for_validation)
            data_fractions = (train_param.exclude_fraction_for_validation,
                              train_param.exclude_fraction_for_inference)

        if data_param:
            data_partitioner.initialise(
                data_param=data_param,
                new_partition=do_new_partition,
                ratios=data_fractions,
                data_split_file=system_param.dataset_split_file)

        if data_param and self.is_training and self.validation_every_n > 0:
            assert data_partitioner.has_validation, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available.\nPlease ' \
                'check dataset partition list {} ' \
                '(remove file to generate a new dataset partition). ' \
                'Or set validation_every_n to -1.'.format(
                    self.validation_every_n, system_param.dataset_split_file)

        # initialise readers}
        #DATA_PARAM: u'abdominal': Namespace(axcodes=(u'A', u'R', u'S'), 
        # csv_file='demos/TrainingAb/file_list.csv', interp_order=3, pixdim=(1.0, 1.0, 1.0),
        # spatial_window_size=(104, 104, 80)),
        # csv_file='demos/TrainingAb/file_list_seg.csv', interp_order=0, pixdim=(1.0, 1.0, 1.0),
        # spatial_window_size=(104, 104, 80))

        #APP_PARAM:  Namespace(image=(u'abdominal',), label=(u'label',), label_normalisation=True,
        # min_numb_labels=2, min_sampling_ratio=1e-06, name='net_segment', num_classes=14, 
        # output_prob=False, sampler=(), weight=())


        self.app.initialise_dataset_loader(
            data_param, app_param, data_partitioner)

        self._data_partitioner = data_partitioner

        # pylint: disable=not-context-manager
        with self.graph.as_default(), tf.name_scope('Sampler'):
            self.app.initialise_sampler()

    def run_application(self):
        """
        Initialise a TF graph, connect data sampler and network within
        the graph context, run training loops or inference loops.

        The training loop terminates when ``self.final_iter`` reached.
        The inference loop terminates when there is no more
        image sample to be processed from image reader.

        :return:
        """
        config = ApplicationDriver._tf_config()
       
        with tf.Session(config=config, graph=self.graph) as session:

            tf.logging.info('Filling queues (this can take a few minutes)')
            self._coord = tf.train.Coordinator()

            # start samplers' threads
            try:
                samplers = self.app.get_sampler()
                if samplers is not None:
                    all_samplers = [s for sets in samplers for s in sets]
                    for sampler in all_samplers:
                        sampler.run_threads(
                            session, self._coord, self.num_threads)
            except (TypeError, AttributeError, IndexError):
                tf.logging.fatal(
                    "samplers not running, pop_batch_op operations "
                    "are blocked.")
                raise

            self.graph = self._create_graph(self.graph)
            self.app.check_initialisations()

            # initialise network
            # fill variables with random values or values from file
            tf.logging.info('starting from iter %d', self.initial_iter)
            self._rand_init_or_restore_vars(session)


            #print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWEIGHT_INIIII. ", weight_ini)

            start_time = time.time()
            loop_status = {}
            #WEIGHTS = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            WEIGHTS = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('b:0')]
        

            print("WEIGHTTTTTTTTTTTTS VARIABLES: ", WEIGHTS)
            print("FINNN WEIGHTTTTTTTTTTTTS VARIABLES")

            
            try:
                # iteratively run the graph
                if self.is_training:                    
                    loop_status['current_iter'] = self.initial_iter
                    self._training_loop(session, loop_status)

                    #print("MMMMMMMMMMMMMMMMMMMMEssage, ", self.MESSAGE)
                    collected = self.outputs_collector
                    #print("CCCCCCCCCCCCCCCCCcollected variables NOUTPUT: ", collected.variables(NETWORK_OUTPUT))
                    #print("NOUTput eval, ", collected.variables(NETWORK_OUTPUT).eval())
                    #print("CCCCCCCCCCCCCCCCCcollected variables CONSOLE: ", collected.variables(CONSOLE))
                    
                    #self.GROUNDTRUTH, self.PREDICTION, self.CONT = self.app.return_seg_args()                                                                               

                    #print("INI_WEIGHTTTTTTTTTTT", self.INI_WEIGHT)                    
                    

                    #print("Showing first slice shape: ", first_slice_shape)
                    #print("number in range; ", self.CONT.eval())
                    #print("groundtruth", self.GROUNDTRUTH.eval())
                    #print("PREDICTION: ", self.PREDICTION.eval())
                    #print("Showing first slice vector shape: ", first_slice_vector.shape)
                    #print("Showing netOut shape: ", netOut_shape) 

                    #------------------------------

                else:
                    loop_status['all_saved_flag'] = False
                    self._inference_loop(session, loop_status)            

            except KeyboardInterrupt:
                tf.logging.warning('User cancelled application')
            except tf.errors.OutOfRangeError:
                pass
            except RuntimeError:
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            finally:
                tf.logging.info('Cleaning up...')
                
                #if self.is_training and loop_status.get('current_iter', None):
                #    self._save_model(session, loop_status['current_iter'])
                #elif not loop_status.get('all_saved_flag', None):
                #    tf.logging.warning('stopped early, incomplete loops')
                
                #iter malogradas: 24999 en adelante

                tf.logging.info('stopping sampling threads')
                self.app.stop()
                tf.logging.info(
                    "%s stopped (time in second %.2f).",
                    type(self.app).__name__, (time.time() - start_time))                     

          

    # pylint: disable=not-context-manager
    def _create_graph(self, graph=tf.Graph()):
        """
        TensorFlow graph is only created within this function.
        """
        assert isinstance(graph, tf.Graph)
        main_device = self._device_string(0, is_worker=False)
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):

            # initialise network, these are connected in
            # the context of multiple gpus
            self.app.initialise_network()
            self.app.add_validation_flag()

            # for data parallelism --
            #     defining and collecting variables from multiple devices
            bn_ops = None
            for gpu_id in range(0, max(self.num_gpus, 1)):
                worker_device = self._device_string(gpu_id, is_worker=True)
                scope_string = 'worker_{}'.format(gpu_id)
                with tf.name_scope(scope_string) as scope:
                    with tf.device(worker_device):
                        # setup network for each of the multiple devices
                        self.app.connect_data_and_network(
                            self.outputs_collector,
                            self.gradients_collector)
                        if self.is_training:
                            # batch norm statistics from the last device
                            bn_ops = tf.get_collection(BN_COLLECTION, scope)

            # assemble all training operations
            if self.is_training and self.gradients_collector:
                updates_op = []
                # batch normalisation moving averages operation
                if bn_ops:
                    updates_op.extend(bn_ops)
                # combine them with model parameter updating operation
                with tf.name_scope('ApplyGradients'):
                    with graph.control_dependencies(updates_op):
                        self.app.set_network_gradient_op(
                            self.gradients_collector.gradients)

            # initialisation operation
            with tf.name_scope('Initialization'):
                self._init_op = global_vars_init_or_restore()

            with tf.name_scope('MergedOutputs'):
                self.outputs_collector.finalise_output_op()
            # saving operation
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints)

        # no more operation definitions after this point
        tf.Graph.finalize(graph)
        return graph

    def _rand_init_or_restore_vars(self, sess):
        """
        Randomly initialising all trainable variables defined in session,
        or loading checkpoint files as variable initialisations.
        """
        if self.is_training and self.initial_iter == 0:
            sess.run(self._init_op)
            tf.logging.info('Parameters from random initialisations ...')
            return
        # check model's folder
        assert os.path.exists(self.model_dir), \
            "Model folder not found {}, please check" \
            "config parameter: model_dir".format(self.model_dir)

        # check model's file
        ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt_state is None:
            tf.logging.warning(
                "%s/checkpoint not found, please check "
                "config parameter: model_dir", self.model_dir)
        if self.initial_iter > 0:
            checkpoint = '{}-{}'.format(self.session_prefix, self.initial_iter)
        else:
            try:
                checkpoint = ckpt_state.model_checkpoint_path
                assert checkpoint, 'checkpoint path not found ' \
                                   'in {}/checkpoints'.format(self.model_dir)
                self.initial_iter = int(checkpoint.rsplit('-')[-1])
                tf.logging.info('set initial_iter to %d based '
                                'on checkpoints', self.initial_iter)
            except (ValueError, AttributeError):
                tf.logging.fatal(
                    'failed to get iteration number '
                    'from checkpoint path, please set '
                    'inference_iter or starting_iter to a positive integer')
                raise
        # restore session
        tf.logging.info('Accessing %s ...', checkpoint)
        try:
            self.saver.restore(sess, checkpoint)
        except tf.errors.NotFoundError:
            tf.logging.fatal(
                'checkpoint %s not found or variables to restore do not '
                'match the current application graph', checkpoint)
            raise
    def run_vars(self, sess, message):
        """
        Running a TF session by retrieving variables/operations to run,
        along with data for feed_dict.

        This function sets ``message._current_iter_output`` with session.run
        outputs.
        """
        # update iteration status before the batch process
        self.app.set_iteration_update(message)

        collected = self.outputs_collector        
        # building a dictionary of variables
        vars_to_run = message.ops_to_run        
        if message.is_training:
            # always apply the gradient op during training
            vars_to_run['gradients'] = self.app.gradient_op

        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$vars_to_run[gradients]: ",self.app.gradient_op)
        # session will run variables collected under CONSOLE
        vars_to_run[CONSOLE] = collected.variables(CONSOLE)
        # session will run variables collected under NETWORK_OUTPUT
        vars_to_run[NETWORK_OUTPUT] = collected.variables(NETWORK_OUTPUT)
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$vars_to_run[NET_OUT]: ",collected.variables(NETWORK_OUTPUT))        
        if self.is_training and self.tensorboard_every_n > 0 and \
                (message.current_iter % self.tensorboard_every_n == 0):
            # session will run variables collected under TF_SUMMARIES
            vars_to_run[TF_SUMMARIES] = collected.variables(TF_SUMMARIES)

        # run the session
        graph_output = sess.run(vars_to_run, feed_dict=message.data_feed_dict)

        # outputs to message
        message.current_iter_output = graph_output

        # update iteration status after the batch process
        # self.app.set_iteration_update(message)
    def _training_loop(self, sess, loop_status):
        '''
        At each iteration, an ``IterationMessage`` object is created
        to send network output to/receive controlling messages from self.app.
        The iteration message will be passed into `self.run_vars`,
        where graph elements to run are collected and feed into `session.run()`.
        A nested validation loop will be running
        if self.validation_every_n > 0.  During the validation loop
        the network parameters remain unchanged.
        '''

        iter_msg = IterationMessage()


        # initialise tf summary writers
        writer_train = tf.summary.FileWriter(
            os.path.join(self.summary_dir, TRAIN), sess.graph)
        writer_valid = tf.summary.FileWriter(
            os.path.join(self.summary_dir, VALID), sess.graph) \
            if self.validation_every_n > 0 else None

        for iter_i in range(self.initial_iter, self.final_iter):
            # general loop information
            loop_status['current_iter'] = iter_i
            if self._coord.should_stop():
                break
            if iter_msg.should_stop:
                break

            # update variables/operations to run, from self.app
            iter_msg.current_iter, iter_msg.phase = iter_i, TRAIN
            self.run_vars(sess, iter_msg)           



            self.app.interpret_output(
                iter_msg.current_iter_output[NETWORK_OUTPUT])
            iter_msg.to_tf_summary(writer_train)
            tf.logging.info(iter_msg.to_console_string())
            print("In  training inside app driver***")
            # run validations if required
            if iter_i > 0 and self.validation_every_n > 0 and \
                    (iter_i % self.validation_every_n == 0):
                for _ in range(self.validation_max_iter):
                    iter_msg.current_iter, iter_msg.phase = iter_i, VALID
                    
                    self.MESSAGE = iter_msg
                    self.run_vars(sess, iter_msg)

                    # save iteration results
                    if writer_valid is not None:
                        iter_msg.to_tf_summary(writer_valid)
                    tf.logging.info(iter_msg.to_console_string())

            print("Number of iter", iter_i)
            print("Module: ", iter_i % self.save_every_n )
            if self.save_every_n > 0 and (iter_i % self.save_every_n == 0):
                #print("SAVE EVERYYYY: ",self.save_every_n)
                self._save_model(sess, iter_i)

            ####printing segmentation application variables
            
            self.loss_variable = self.app.return_loss_variable()
            SUMA = self.app.SUMA
            #print("My print: loss", self.loss_variable.eval())
            #print("Suma de tensorss, ", SUMA.eval())
            #lossVariable = self.app.return_loss_variable()
            first_slice, first_slice_shape, netOut, netOut_shape = self.app.return_first_slice()
            first_slice_vector = first_slice.eval()
            
            grad = self.app.GRADS


            GRADS = self.app.GRADS
            
            
            print("PPPPPPPPPpPrinting GRADDSSS")
            for g, v in GRADS:
                if g is not None:
                    if v.name == "DenseVNet/conv_1/conv_/w:0":
                        print("HHHHHHHHHHoliiii from grads")
                        print("Name: ", v.name, "")                       
                        print("Printing v eval: ", " - ", v.shape,": ",  v.eval())
                        print("****************this is gradient*************")
                        print("gradient's name: ",g.name," - ",g.shape)
                        print("gradient name: ", g.eval())                
                    
            print("OUT of GRADSSS")

            '''
            CONVV = self.graph.get_tensor_by_name('worker_0/DenseVNet/conv_1/conv_/CONVV:0').eval()
            #print("CONVVVV: ", CONVV)

            CONVV2 = self.graph.get_tensor_by_name('worker_0/DenseVNet/dense_feature_stack_block_2/conv_bn_relu/conv_/CONVV:0').eval()
            #print("CONVVVV-1: ", CONVV2)

            PREDICTION = self.graph.get_tensor_by_name('worker_0/loss_function/Reshape_1:0')
            GT = self.graph.get_tensor_by_name('worker_0/loss_function/Reshape:0')
            
                
            #print("PPPPREDICTION: ", PREDICTION, " - ", PREDICTION.eval())
            #print("GGGGGGGGGt: ", GT, " - ", GT.eval())

            Entropy = self.graph.get_tensor_by_name('worker_0/loss_function/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0')
            print("EEEEEntropy", Entropy, "- ", Entropy.eval())    

            Reduced_entropy = self.graph.get_tensor_by_name('worker_0/loss_function/Mean_1:0')
            print("RRRRRRreduced entropy: ", Reduced_entropy)
            print(" - ", Reduced_entropy.eval())

            '''
            '''
            LAST_CONV_WEIGHTS = self.graph.get_tensor_by_name('DenseVNet/conv_1/conv_/w:0')
            print("llllllAST CONV WEIGHTS, ", LAST_CONV_WEIGHTS,": ",LAST_CONV_WEIGHTS.shape)
            print(" - ", LAST_CONV_WEIGHTS.eval())
            '''
            
            #BDO = self.graph.get_tensor_by_name('worker_0/DenseVNet/conv_1/MYOUTPUTTENSOR:0')
            #print("BBBDO: ", BDO.eval())
            #ADO = self.graph.get_tensor_by_name('worker_0/DenseVNet/conv_1/MYOUTPUTTENSOR_DROPOUT:0')
            #print("AAADO: ", ADO.eval())
            #print("#######################################GRADS", grad)  
            '''
            grad = self.app.GRADS
            grad2 = self.app.GRADS[0]
            print("#######################################GRADS", grad)  
            print("#######################################GRADS2",  grad2)
            '''
            #print("trainable variables: ", tf.trainable_variables())
            print("SSSSSSSSSSSSSSSSSsaliendo del training loop")
            ##################################
            ############################
            #################                                          
    def _inference_loop(self, sess, loop_status):
        """
        Runs all variables returned by outputs_collector,
        this loop stops when the return value of
        application.interpret_output is False.
        """
        iter_msg = IterationMessage()
        loop_status['all_saved_flag'] = False
        iter_i = 0
        while True:
            if self._coord.should_stop():
                break
            if iter_msg.should_stop:
                break

            iter_msg.current_iter, iter_msg.phase = iter_i, INFER
            # run variables provided in `iter_msg` and set values of
            # variables to iter_msg.current_iter_output
            self.run_vars(sess, iter_msg)
            iter_i = iter_i + 1

            # process the graph outputs
            if not self.app.interpret_output(
                    iter_msg.current_iter_output[NETWORK_OUTPUT]):
                tf.logging.info('processed all batches.')
                loop_status['all_saved_flag'] = True
                break
            tf.logging.info(iter_msg.to_console_string())          
    def _save_model(self, session, iter_i):
        """
        save session parameters to the hard drive
        """
        if iter_i <= 0:
            return
        self.saver.save(sess=session,
                        save_path=self.session_prefix,
                        global_step=iter_i)
        tf.logging.info('iter %d saved: %s', iter_i, self.session_prefix)
    def _device_string(self, device_id=0, is_worker=True):
        """
        assigning CPU/GPU based on user specifications
        """
        # pylint: disable=no-name-in-module
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        n_local_gpus = sum([x.device_type == 'GPU' for x in devices])
        if self.num_gpus <= 0:  # user specified no gpu at all
            return '/cpu:{}'.format(device_id)
        if self.is_training:
            # in training: use gpu only for workers whenever n_local_gpus
            device = 'gpu' if (is_worker and n_local_gpus > 0) else 'cpu'
            if device == 'gpu' and device_id >= n_local_gpus:
                tf.logging.fatal(
                    'trying to use gpu id %s, but only has %s GPU(s), '
                    'please set num_gpus to %s at most',
                    device_id, n_local_gpus, n_local_gpus)
                raise ValueError
            return '/{}:{}'.format(device, device_id)
        # in inference: use gpu for everything whenever n_local_gpus
        return '/gpu:0' if n_local_gpus > 0 else '/cpu:0'
    @staticmethod
    def _console_vars_to_str(console_dict):
        """
        Printing values of variable evaluations to command line output
        """
        if not console_dict:
            return ''
        console_str = ', '.join(
            '{}={}'.format(key, val) for (key, val) in console_dict.items())
        return console_str
    @staticmethod
    def _create_app(app_type_string):
        """
        Import the application module
        """
        return ApplicationFactory.create(app_type_string)
    @staticmethod
    def _tf_config():
        """
        tensorflow system configurations
        """
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True
        return config
