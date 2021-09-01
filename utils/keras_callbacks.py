import wandb
import tensorflow as tf


class WandbCallback(tf.keras.callbacks.Callback):

    def __init__(self, wandb_run, train_data):
        super(WandbCallback, self).__init__()
        self.train_data = train_data
        self.wandb_run = wandb_run

    @tf.function
    def _get_gradient(self, x, y, sw):
        # calculate gradients
        with tf.GradientTape() as tape:
            y_p = self.model(x)
            loss = self.model.loss(y, y_p, sw)
        grad = tape.gradient(loss, self.model.trainable_variables)
        return grad

    def on_epoch_end(self, epoch, logs):
        # log metrics
        for key in logs.keys():
            self.wandb_run.log({key: logs[key]}, step=epoch)
        # log lr
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.wandb_run.log({'lr': float(lr)}, step=epoch)
        # get gradients
        grad = self._get_gradient(*next(iter(self.train_data)))
        # log gradients and weights
        for nlayer in range(len(self.model.trainable_variables)):
            layer = self.model.trainable_variables[nlayer]
            self.wandb_run.log({
                'weight_'+layer.name: wandb.Histogram(layer.numpy()),
                'gradient_'+layer.name: wandb.Histogram(grad[nlayer].numpy())
            }, step=epoch)


class WandbCallback_distribute(WandbCallback):

    def __init__(self, wandb_run, batch_size, train_data, strategy):
        super(WandbCallback_distribute, self).__init__(wandb_run, train_data)
        self.strategy = strategy
        self.global_batch_size = batch_size
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _cal_gradient(self, x, y, sw):
        with tf.GradientTape() as tape:
            y_p = self.model(x)
            per_example_loss = self.loss_object(y, y_p, sw)
            loss = tf.nn.compute_average_loss(
                per_example_loss,
                global_batch_size=self.global_batch_size
            )
        grad = tape.gradient(loss, self.model.trainable_variables)
        return grad

    @tf.function
    def _get_gradient(self, x, y, sw):
        grad = self.strategy.run(self._cal_gradient, args=(x, y, sw))
        grad = [tf.stack(grad[n].values, 0) for n in range(len(grad))]
        return grad

