import numpy as np
import tensorflow as tf

def differential_equation_loss(model_u, x, t):
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(t)
            u = model_u(x, t)
        du_dt = tape2.gradient(u, t)
        du_dx = tape2.gradient(u, x)
    du2_dx2 = tape1.gradient(du_dx, x)
    coefficient = 0.01 / np.pi
    f = du_dt + u * du_dx - coefficient * du2_dx2   # burgers equation
    del tape2
    return f

def optimizer_function_factory(model, loss, input_u, output_u, input_f):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """
    x_u = input_u[:, 0:1]
    t_u = input_u[:, 1:2]
    x_f = input_f[:, 0:1]
    t_f = input_f[:, 1:2]
    model(x_u,t_u)
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_weights)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_weights[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(output_u, model(x_u,t_u), differential_equation_loss(model, x_f, t_f))

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    return f