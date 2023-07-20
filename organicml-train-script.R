library(keras)
library(tensorflow)

train_dir <- "data/train"
validation_dir <- "data/validation"

create_generators <- function(test_validation_batch_size) {
  validation_datagen <- image_data_generator(rescale = 1/255)
  train_datagen <- image_data_generator(rescale = 1/255)
  
  train_generator <- flow_images_from_directory(
    train_dir,
    train_datagen,
    target_size = c(256, 256),
    batch_size = test_validation_batch_size,
    shuffle = FALSE,
    class_mode = "binary"
  )
  
  validation_generator <- flow_images_from_directory(
    validation_dir,
    validation_datagen,
    target_size = c(256, 256),
    batch_size = test_validation_batch_size,
    shuffle = FALSE,
    class_mode = "binary"
  )
  
  list(train_generator = train_generator, validation_generator = validation_generator)
}

create_network <- function() {
  conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = c(256, 256, 3)
  )
  
  conv_base %>% freeze_weights()
  
  network <- keras_model_sequential() %>%
    conv_base %>%
    layer_flatten() %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  network %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(learning_rate = 0.5e-5),
    metrics = c("accuracy")
  )
  
  network
}

convnet_1 <- create_network()
summary(convnet_1)

tensorflow::set_random_seed(0)

callbacks_list <- list(
  callback_model_checkpoint(
    filepath = "organicml_checkpoint.h5",
    monitor = "val_accuracy",
    mode = "max",
    save_best_only = TRUE
  )
)

test_validation_batch_size <- 25
generators_1 <- create_generators(test_validation_batch_size = test_validation_batch_size)

fit_history_1 <- convnet_1 %>%
  fit(
    generators_1$train_generator,
    steps_per_epoch = 300/test_validation_batch_size,
    epochs = 50,
    validation_data = generators_1$validation_generator,
    validation_steps = 110/test_validation_batch_size,
    callbacks = callbacks_list
  )

max_val_accuracy_1 <- max(fit_history_1$metrics$val_accuracy)
argmax_val_accuracy_1 <- which.max(fit_history_1$metrics$val_accuracy)
cat("Maximum val_accuracy: ", max_val_accuracy_1, "\n")
cat("Epoch of maximum validation accuracy: ", argmax_val_accuracy_1, "\n")
