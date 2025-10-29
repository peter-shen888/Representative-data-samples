
def lr_schedule(epoch):
 if epoch < 600:
  return 0.0001  # Learning rate for the first 500 epochs
 elif epoch < 1000:
  return 0.00005  # Learning rate for the next 500 epochs
 elif epoch < 1200:
  return 0.00002  # Learning rate for the next 500 epochs
 else:
  return 0.00001  # Learning rate for epochs beyond 5000


class PhysicsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        P_loss, P_generator, P_load, P_transmission = inputs
        P_generator_sum = tf.reduce_sum(P_generator)
        P_load_sum = tf.reduce_sum(P_load)
        return - P_load_sum + P_generator_sum - P_loss + P_transmission


weights_P_load = tf.constant([1022.726867, 756.3148956, 154.0702019, 266.3509752, 546.2137298, 588.2009735, 1290.781464], dtype=tf.float32)  # Replace a1, a2, ..., a7 with actual weights
biases_P_load = tf.constant([292.2015686, 216.0866547, 44.01995468, 76.09980775, 156.0592804, 168.0550537, 368.7834168], dtype=tf.float32)  # Replace b1, b2, ..., b7 with actual biases
# Apply the linear transformation to the first 7 features
transformed_P_load = decoded[:, :7] * weights_P_load + biases_P_load
# Sum the transformed features
P_load = tf.reduce_sum(transformed_P_load, axis=1)


physics_output_P = PhysicsLayer()([P_loss, P_generator, P_load, P_transmission]) * 0.00001
physics_output_Q = ReactivePhysicsLayer()([Q_loss, Q_generator, Q_load, Q_transmission]) * 0.0001


autoencoder = tf.keras.Model(inputs=input_layer, outputs=[decoded, physics_output_P, physics_output_Q])

lambda_P = 0.1
lambda_Q = 0.1


def custom_loss(y_true, y_pred):

    decoded_pred = y_pred
    physics_output_P_pred = y_pred
    physics_output_Q_pred = y_pred


    decoded_true = y_true
    physics_output_P_true = y_true
    physics_output_Q_true = y_true


    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(decoded_true, decoded_pred))


    physics_loss_P = tf.reduce_mean(tf.square(physics_output_P_pred - physics_output_P_true))
    physics_loss_Q = tf.reduce_mean(tf.square(physics_output_Q_pred - physics_output_Q_true))

    total_loss = reconstruction_loss + lambda_P * physics_loss_P + lambda_Q * physics_loss_Q
    return total_loss