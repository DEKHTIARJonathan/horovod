# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import horovod.tensorflow as hvd
from horovod.tensorflow import Compression

from distutils.version import LooseVersion


__all__ = [
    "create_distributed_optimizer",
    "broadcast_global_variables",
    "allgather",
    "allreduce",
    "broadcast",
    "load_model",
]


def create_distributed_optimizer(keras):

    from keras import backend as K
    from keras.legacy import interfaces

    class DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, optimizer, name=None, device_dense='', device_sparse='',
                     compression=Compression.none, sparse_as_dense=False):
            """An optimizer that wraps another keras.optimizers.Optimizer, using an allreduce to
            average gradient values before applying gradients to model weights.

            Args:
                optimizer: Optimizer to use for computing gradients and applying updates.
                name: Optional name prefix for the operations created when applying
                      gradients. Defaults to "Distributed" followed by the provided
                      optimizer type.
                device_dense: Device to be used for dense tensors. Uses GPU by default
                              if Horovod was build with HOROVOD_GPU_ALLREDUCE.
                device_sparse: Device to be used for sparse tensors. Uses GPU by default
                               if Horovod was build with HOROVOD_GPU_ALLGATHER.
                compression: Compression algorithm used to reduce the amount of data
                             sent and received by each worker node.  Defaults to not
                             using compression.
                sparse_as_dense: Treat all sparse gradients as dense tensors.  This can
                                 help improve performance and memory utilization if
                                 the original sparse gradient has high density.
                                 Defaults to false.
            """

            if not isinstance(optimizer, keras.optimizers.Optimizer):
                raise ValueError('"optimizer" must be an instance of `keras.optimizers.Optimizer`, but '
                                 'got: %s' % optimizer)

            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__

            self._name = name
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self._get_gradients_used = False
            self._optimizer = optimizer

            try:
                self._track_trackable(self._optimizer, 'base_optimizer')
            except AttributeError:
                pass  # Does not exist in native Keras

            # Needed because the superclass's __getattribute__ checks this.
            self._hyper = {}

        def _compute_gradients(self, loss, var_list, grad_loss=None):
            loss = self.get_scaled_loss(loss)
            grads_and_vars = self._optimizer._compute_gradients(loss, var_list,  # pylint: disable=protected-access
                                                                grad_loss)
            grads = [g for g, _ in grads_and_vars]
            variables = [v for _, v in grads_and_vars]
            unscaled_grads = self.get_unscaled_gradients(grads)
            return list(zip(unscaled_grads, variables))

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            self._get_gradients_used = True

            gradients = self._optimizer.get_gradients(loss, params)

            if hvd.size() > 1:
                averaged_gradients = []
                with tf.name_scope(self._name + "_Allreduce"):
                    for grad in gradients:
                        if grad is not None:
                            if self._sparse_as_dense and \
                                    isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            avg_grad = hvd.allreduce(grad,
                                                     device_dense=self._device_dense,
                                                     device_sparse=self._device_sparse,
                                                     compression=self._compression)
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    return averaged_gradients
            else:
                return gradients

        def apply_gradients(self, grads_and_vars, name=None):

            if not self._get_gradients_used:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()`. If you\'re using TensorFlow 2.0, '
                                'please specify `experimental_run_tf_function=False` in '
                                '`compile()`.')

            return self._optimizer.get_gradients(grads_and_vars, name)

        def get_config(self):

            return {
                'name': self._name,
                'device_dense': self._device_dense,
                'device_sparse': self._device_sparse,
                'compression': self._compression,
                'sparse_as_dense': self._sparse_as_dense,
                'optimizer': keras.optimizers.serialize(self._optimizer),
            }

        @classmethod
        def from_config(cls, config, custom_objects=None):
            print(config['optimizer'])

            import copy
            config = copy.deepcopy(config)  # Make a copy, since we mutate config
            config['optimizer'] = keras.optimizers.deserialize(
                config['optimizer'],
                custom_objects=custom_objects
            )

            print(config['optimizer'])

            return cls(**config)

        # Delegations: We delegate most OptimizerV2 methods to the wrapped optimizer
        # below.

        @property
        def iterations(self):
            return self._optimizer.iterations

        @iterations.setter
        def iterations(self, variable):
            self._optimizer.iterations = variable

        def get_slot_names(self):
            return self._optimizer.get_slot_names()

        def variables(self):
            return self._optimizer.variables()

        @property
        def weights(self):
            return self._optimizer.weights

        def get_weights(self):
            return self._optimizer.get_weights()

        def set_weights(self, weights):
            return self._optimizer.set_weights(weights)

        # For the most part, we only expose methods in the base OptimizerV2, not
        # individual subclasses like Adam. However, although "learning_rate" and "lr"
        # properties are not part of the base OptimizerV2 class, they are part of most
        # subclasses, so we expose them here for convenience.

        @property
        def learning_rate(self):
            return self._optimizer.learning_rate

        @learning_rate.setter
        def learning_rate(self, lr):
            self._optimizer.learning_rate = lr

        @property
        def lr(self):
            return self._optimizer.lr

        @lr.setter
        def lr(self, lr):
            self._optimizer.lr = lr

        @interfaces.legacy_get_updates_support
        @K.symbolic
        def get_updates(self, loss, params):
            return self._optimizer.get_updates(loss, params)

        def get_slot(self, var, slot_name):
            # We cannot implement get_slot for the following reason: When saving a
            # checkpoint, two optimizers cannot share slot variables. Since both the
            # DistributedOptimizer and the wrapped optimizer (self and self._optimizer
            # respectively) are checkpointed, we cannot expose the wrapped optimizer's
            # slots in the DistributedOptimizer. Otherwise, a checkpoint would believe
            # both optimizers share slot variables.
            raise AttributeError(
                'You cannot call get_slot on a DistributedOptimizer. This limitation '
                'will be removed in the future.')

        def add_slot(self, var, slot_name, initializer='zeros'):
            # We disallow adding a slot for consistency with `get_slot`.
            raise AttributeError(
                'You cannot call add_slot on a DistributedOptimizer. This limitation '
                'will be removed in the future.')

        # We do not override some OptimizerV2 methods. For each, we describe why we do
        # not delegate them to self._optimizer:
        # * get_updates: get_updates() calls get_gradients(). Since we override
        #   get_gradients(), we cannot delegate get_updates() to self._optimizer,
        #   otherwise the overridden get_gradients() method would not be called.
        #   Luckily, get_updates() does not access any OptimizerV2 fields, so
        #   inheriting the OptimizerV2 version works fine.
        # * minimize: We don't delegate for a similar as get_updates(): it calls
        #   both self._compute_gradients() and self.apply_gradients(), and both need
        #   to have the DistributedOptimizer version called.

        # TODO(reedwm): Maybe merge this class's functionality into OptimizerV2.

        # TODO(reedwm): Maybe throw an error if mixed precision is used without this
        # optimizer being used.

    from keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
    from keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
    keras.utils.get_custom_objects()[DistributedOptimizer.__name__] = DistributedOptimizer

    return DistributedOptimizer


def _eval(backend, op_or_result):
    if hvd._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(hvd, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, hvd.broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average):
    return _eval(backend, hvd.allreduce(tf.constant(value, name=name), average=average))


def allgather(backend, value, name):
    return _eval(backend, hvd.allgather(tf.constant(value, name=name)))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, hvd.broadcast(tf.constant(value, name=name), root_rank))


def load_model(keras, filepath, custom_optimizers, custom_objects):
    # horovod_objects = {
    #     subclass.__name__.lower(): wrap_optimizer(subclass)
    #     for subclass in keras.optimizers.Optimizer.__subclasses__()
    #     if subclass.__module__ == keras.optimizers.Optimizer.__module__
    # }
    horovod_objects = dict()

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: cls
            for cls in custom_optimizers
        })

    print(horovod_objects)

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    print(horovod_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
