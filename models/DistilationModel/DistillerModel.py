from tensorflow import keras

class Distiller(keras.Model):

    # Needs both the student and teacher models to create an instance of this class
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    # Will be used when calling model.compile()
    def compile(self, optimizer, metrics, student_loss_fn,
                distillation_loss_fn, alpha, temperature):
        # Compile using the optimizer and metrics
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)

        # Add the other params to the instance
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    # Will be used when calling model.fit()
    def train_step(self, data):
        # Data is expected to be a tuple of (features, labels)
        x, y = data

        # Vanilla forward pass of the teacher
        # Note that the teacher is NOT trained
        teacher_predictions = self.teacher(x, training=False)

        # Use GradientTape to save gradients
        with tf.GradientTape() as tape:
            # Vanilla forward pass of the student
            student_predictions = self.student(x, training=True)

            # Compute vanilla student loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute distillation loss
            # Should be KL divergence between logits softened by a temperature factor
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1))

            # Compute loss by weighting the two previous losses using the alpha param
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Use tape to calculate gradients for student
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update student weights
        # Note that this done ONLY for the student
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a performance dictionary
        # You will see this being outputted during training
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    # Will be used when calling model.evaluate()
    def test_step(self, data):
        # Data is expected to be a tuple of (features, labels)
        x, y = data

        # Use student to make predictions
        # Notice that the training param is set to False
        y_prediction = self.student(x, training=False)

        # Calculate student's vanilla loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a performance dictionary
        # You will see this being outputted during inference
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
