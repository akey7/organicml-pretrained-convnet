library(keras)
library(tibble)
library(dplyr)
library(readr)

batch_size <- 10
validation_dir <- "data/validation"

validation_datagen <- image_data_generator(rescale = 1/255)

validation_generator <- flow_images_from_directory(
    validation_dir,
    validation_datagen,
    target_size = c(256, 256),
    batch_size = batch_size,
    shuffle = FALSE,
    class_mode = "binary"
  )

network <- load_model_hdf5("organicml_checkpoint.h5")
network %>% evaluate(validation_generator)

predicted <- c()
actual <- c()
network_input <- c()
i <- 0
while(TRUE) {
  batch <- generator_next(validation_generator)
  inputs_batch <- batch[[1]]
  labels_batch <- batch[[2]]
  predicted <- c(predicted, round(network %>% predict(inputs_batch)))
  actual <- c(actual, labels_batch)
  network_input <- c(network_input, inputs_batch)
  i <- i + 1
  if (i * batch_size >= validation_generator$samples)
    break
}

evaluation_df <- tibble(
    image_filename = validation_generator$filepaths,
    actual = actual,
    predicted = predicted
  )

network_performance <- evaluation_df %>%
  summarize(
    accuracy = sum(actual == predicted) / n(),
    tpr = sum(actual == 1 & predicted == 1) / sum(actual == 1),
    tnr = sum(actual == 0 & predicted == 0) / sum(actual == 0),
    fpr = sum(actual == 0 & predicted == 1) / sum(actual == 0),
    fnr = sum(actual == 1 & predicted == 0) / sum(actual == 1)
  )

evaluation_df %>%
  filter(actual != predicted) %>%
  write_csv("organicml-evaluation-failures.csv")
